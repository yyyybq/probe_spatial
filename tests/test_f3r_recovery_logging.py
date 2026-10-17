import ast
import copy
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def parse(path):
    tree=ast.parse(path.read_text())
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'encode_images_end_time - encode_images_start_time > 20'==ast.unparse(n.test))
    return tree,block


class LoggingTests(unittest.TestCase):
    def test_only_diagnostics_changed(self):
        old,oldblock=parse(ROOT/'features/f3r/f3r/models/fast3r.py')
        new,newblock=parse(ROOT/'runtime/f3r_qwen_recovery_v1/source/features/f3r/f3r/models/fast3r.py')
        oldblock.body=copy.deepcopy(newblock.body)
        self.assertEqual(ast.dump(old),ast.dump(new))

    def test_inference_view_slow_branch(self):
        _,block=parse(ROOT/'runtime/f3r_qwen_recovery_v1/source/features/f3r/f3r/models/fast3r.py')
        class Image:
            shape=(1,3,288,512)
            def __repr__(self): raise AssertionError('must not stringify tensor')
        env={'views':[{'img':Image()}],'encode_images_start_time':0,'encode_images_end_time':30,'print':lambda *args:None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=block.body,type_ignores=[])),'<slow-branch>','exec'),env)


if __name__=='__main__':unittest.main()
