"""Writing that survives being killed, and work that resumes where it stopped.

Assume the session dies. On free Kaggle it will: a batch job is hard-killed at
twelve hours, and a kill skips your `finally` block, so anything you were going
to save at the end is simply gone.

Two tools, both small:

- `atomic_write` for files written once, whole.
- `JsonlSink` for long jobs that produce records one batch at a time. It appends
  as it goes and can tell you which items it already holds, so re-running the
  same command finishes the job instead of starting it over.

Run the self-check:  python ml/runner.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write(path, text: str, encoding: str = "utf-8") -> Path:
    """Write to a temporary file beside the target, then rename onto it.

    Renaming within one directory is atomic on every OS this runs on, so a reader
    sees either the whole old file or the whole new one -- never half of either.
    Writing straight over the only copy is how a config or a result file ends up
    truncated when the process is killed at the wrong instant.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


class JsonlSink:
    """An append-as-you-go JSONL file that knows what it already contains.

    The pattern this exists for:

        sink = JsonlSink("data/probe.jsonl")
        for batch in chunks(sink.pending(questions), 16):
            sink.write(run_the_expensive_thing(batch))

    Kill it at any point and re-run the same line. `pending` filters out
    everything already on disk, so the work resumes instead of restarting.

    A kill can also land mid-line, leaving a final fragment that is not valid
    JSON. Rather than crashing on the next read -- or worse, silently skipping it
    forever -- the truncated tail is dropped and the file rewritten atomically,
    with the repair reported in `repaired`.
    """

    def __init__(self, path, key: str = "qid"):
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired = 0
        self._done: set = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        kept, bad = [], 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = record[self.key]
            except (json.JSONDecodeError, KeyError):
                bad += 1
                continue
            kept.append(line)
            self._done.add(key)
        if bad:
            atomic_write(self.path, "".join(l + "\n" for l in kept))
            self.repaired = bad

    def write(self, records) -> int:
        """Append records and flush them all the way to disk.

        `fsync` is the part that matters. Without it the lines sit in the OS
        buffer and a hard kill loses them, which defeats the whole point of
        appending as you go.
        """
        records = list(records)
        if not records:
            return 0
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._done.add(record[self.key])
            handle.flush()
            os.fsync(handle.fileno())
        return len(records)

    def pending(self, items):
        """The items not already on disk, in their original order."""
        return [i for i in items if i[self.key] not in self._done]

    def read(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in
                self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def __contains__(self, key) -> bool:
        return key in self._done

    def __len__(self) -> int:
        return len(self._done)


def chunks(items, size: int):
    """Split a list into fixed-size batches. The last one may be short."""
    items = list(items)
    for start in range(0, len(items), size):
        yield items[start:start + size]


def demo():
    """Self-check, including the truncated-file case a kill actually produces."""
    import shutil

    work = Path(tempfile.mkdtemp())
    try:
        target = work / "out.jsonl"
        items = [{"qid": f"q{i}", "question": f"question {i}"} for i in range(6)]

        sink = JsonlSink(target)
        assert len(sink.pending(items)) == 6
        sink.write([{"qid": i["qid"], "answer": "x"} for i in items[:4]])
        assert len(sink) == 4

        # a fresh process opening the same file must see the finished work
        again = JsonlSink(target)
        left = again.pending(items)
        assert [i["qid"] for i in left] == ["q4", "q5"], left
        assert "q0" in again and "q4" not in again

        # simulate a kill mid-write: a half-written final line
        with open(target, "a", encoding="utf-8", newline="\n") as handle:
            handle.write('{"qid": "q4", "ans')
        repaired = JsonlSink(target)
        assert repaired.repaired == 1, "truncated tail should be detected"
        assert len(repaired) == 4, "and dropped, not counted as done"
        assert len(repaired.pending(items)) == 2
        # the repaired file must itself be readable
        assert len(JsonlSink(target).read()) == 4

        # atomic_write leaves no debris and no partial file
        note = work / "deep" / "note.txt"
        atomic_write(note, "hello\n")
        assert note.read_text(encoding="utf-8") == "hello\n"
        atomic_write(note, "replaced\n")
        assert note.read_text(encoding="utf-8") == "replaced\n"
        assert not list(work.rglob("*.tmp")), "temp files left behind"

        assert [len(c) for c in chunks(range(7), 3)] == [3, 3, 1]
        print("runner self-check passed")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    demo()
