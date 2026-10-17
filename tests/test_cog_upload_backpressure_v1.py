import importlib.util
import signal
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


guard = module('guard', ROOT/'runtime/cog_upload_backpressure_v1/guard.py')
ctl = module('ctl', ROOT/'scripts/feature_upload_recovery_control_20260923.py')


class BackpressureTests(unittest.TestCase):
    def test_watermarks(self):
        self.assertTrue(guard.should_run(False, 4*guard.TiB, 20*guard.GiB, True))
        self.assertFalse(guard.should_run(False, 4*guard.TiB, 40*guard.GiB, True))
        self.assertTrue(guard.should_run(True, 4*guard.TiB, 40*guard.GiB, True))
        self.assertFalse(guard.should_run(True, 4*guard.TiB, 65*guard.GiB, True))
        self.assertFalse(guard.should_run(True, guard.TiB, 0, True))
        self.assertFalse(guard.should_run(True, 4*guard.TiB, 0, False))

    def test_exact_frozen_worklist(self):
        import json
        rows=[]
        for job in ('pt-rutyedtm','pt-tkvc8tad','pt-v2ejbno9','pt-3cx4i1b8'):
            saved=json.loads((guard.BASE/'process_control'/job/'paused.json').read_text())
            root, paths=guard.expected_paths(saved['after']['argv'])
            self.assertTrue(str(root).endswith('/cogvideox/scannetpp'))
            rows.extend(paths)
        self.assertEqual(len(rows),len(set(rows)))

    def test_real_pidfd_stop_resume(self):
        import time
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(15)'])
        fd=ctl.pidfd_open(child.pid)
        try:
            identity=ctl.process(child.pid)['start_ticks']
            ctl.pidfd_signal(fd,signal.SIGSTOP)
            for _ in range(50):
                if ctl.process(child.pid)['state']=='T': break
                time.sleep(.01)
            self.assertEqual(ctl.process(child.pid)['state'],'T')
            ctl.pidfd_signal(fd,signal.SIGCONT)
            for _ in range(50):
                if ctl.process(child.pid)['state']!='T': break
                time.sleep(.01)
            self.assertNotEqual(ctl.process(child.pid)['state'],'T')
            self.assertEqual(ctl.process(child.pid)['start_ticks'],identity)
        finally:
            ctl.pidfd_signal(fd,signal.SIGCONT); child.terminate(); child.wait(); __import__('os').close(fd)


if __name__=='__main__': unittest.main()
