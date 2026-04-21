#!/usr/bin/env python3
"""Check all ROOT files in seed_fixed/ for TFile::Recover warnings.
Captures stderr at the file-descriptor level so ROOT's C-level messages
are intercepted.  Prints every file that triggered a recovery and moves
those files to seed_fixed_failed/."""

import os
import shutil
import sys
import threading
import ROOT

ROOT.gROOT.SetBatch(True)

SEED_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_fixed")
FAILED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_fixed_failed")


def open_and_check(fpath):
    """Return (is_zombie, is_recovered, captured_stderr_text)."""
    # Redirect fd 2 (stderr) to a pipe so ROOT's fprintf messages are caught.
    old_fd = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)  # fd 2 is now the write end; close our duplicate

    buf = []

    def _reader():
        while True:
            try:
                chunk = os.read(r, 4096)
                if not chunk:
                    break
                buf.append(chunk.decode("utf-8", errors="replace"))
            except OSError:
                break

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        tf = ROOT.TFile.Open(fpath, "READ")
    finally:
        # Restore real stderr — this closes the write end of the pipe (fd 2
        # is replaced by old_fd), so the reader thread gets EOF and exits.
        os.dup2(old_fd, 2)
        os.close(old_fd)
        t.join(timeout=3)
        os.close(r)

    captured = "".join(buf)
    is_zombie = tf is None or tf.IsZombie()
    is_recovered = "Recover" in captured

    if tf and not tf.IsZombie():
        tf.Close()

    return is_zombie, is_recovered, captured.strip()


def main():
    os.makedirs(FAILED_DIR, exist_ok=True)
    all_files = sorted(f for f in os.listdir(SEED_DIR) if f.endswith(".root"))
    total = len(all_files)
    print(f"Checking {total} ROOT files in seed_fixed/ ...", flush=True)

    zombie_files = []
    recovered_files = []

    for i, fname in enumerate(all_files):
        fpath = os.path.join(SEED_DIR, fname)
        is_zombie, is_recovered, msg = open_and_check(fpath)

        if is_zombie:
            zombie_files.append(fname)
        elif is_recovered:
            recovered_files.append((fname, msg))

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} checked ...", flush=True)

    print(f"\n{'='*60}")
    print(f"Total files checked  : {total}")
    print(f"Zombie / unreadable  : {len(zombie_files)}")
    print(f"TFile::Recover files : {len(recovered_files)}")

    if zombie_files:
        print("\n--- Unreadable / Zombie files ---")
        for f in zombie_files:
            print(f"  {f}")

    if recovered_files:
        print("\n--- Files that needed TFile::Recover ---")
        for fname, msg in recovered_files:
            print(f"\n  {fname}")
            for line in msg.splitlines()[:4]:
                print(f"    {line}")

    # Move bad files to seed_fixed_failed/
    bad_files = zombie_files + [fname for fname, _ in recovered_files]
    if bad_files:
        print(f"\nMoving {len(bad_files)} bad file(s) to seed_fixed_failed/ ...")
        for fname in bad_files:
            src = os.path.join(SEED_DIR, fname)
            dst = os.path.join(FAILED_DIR, fname)
            shutil.move(src, dst)
            print(f"  moved: {fname}")
        print("Done moving.")
    else:
        print("\nNo bad files to move.")

    print("\nDone.")


if __name__ == "__main__":
    main()
