import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

spec = importlib.util.spec_from_file_location('mini_publish', Path(__file__).resolve().parents[1]/'runtime/feature_upload_recovery_v1/minimax_publish.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


class FakeClient:
    def __init__(self, fail_receipt=False, corrupt=False):
        self.values = {}
        self.fail_receipt = fail_receipt
        self.corrupt = corrupt
    def contains(self, uri):
        return uri in self.values
    def get(self, uri):
        return b'corrupt' if self.corrupt else self.values.get(uri)
    def put(self, uri, value):
        if self.fail_receipt and '_upload_receipts_v1/' in uri:
            raise RuntimeError('receipt upload failed')
        self.values[uri] = value


class UploadTests(unittest.TestCase):
    def test_target_arguments_match_original_protocol(self):
        sys.path.insert(0,str(Path(pub.__file__).parent))
        try:
            import minimax_resume as resume
            old=['python','-m','features.run_inscene15k','--vfm','minimax_h3','--num-frames','73',
                 '--prompt','.','--start','0','--end','85','--mode','streaming_prefix',
                 '--out-root','stream','--output-layers','10','20','30','40','49',
                 '--prefix-lengths','8,12,16,24','--prefix-max-len','24']
            target=resume.target_arguments(old)
            self.assertEqual(target[target.index('--num-frames')+1],'73')
            self.assertEqual(target[target.index('--prompt')+1],'.')
            self.assertEqual(target[target.index('--end')+1],'85')
            self.assertIn('--target-from-streaming-windows',target)
            self.assertEqual(target[target.index('--output-layers')+1:target.index('--target-prefix-lengths')],['20','30','40','49'])
            self.assertEqual(target[target.index('--target-horizons')+1],'1,2,4')
            self.assertNotIn('--prefix-max-len',target)
            self.assertEqual(old[old.index('--mode')+1],'streaming_prefix')
        finally:
            sys.path.pop(0)
    def test_cleanup_requires_done_index_and_matching_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); f=root/'f.sft'; f.write_bytes(b'original')
            r={'status':'REMOTE_VERIFIED','route_sha256':pub.ROUTE_SHA,'relative':'f.sft',
               'objects':{'f.sft':{'sha256':pub.sha(b'original')}}}
            index=root/'done'; index.write_text('')
            with self.assertRaises(AssertionError): pub.cleanup_payload(root,'f.sft',r,index)
            index.write_text('f.sft\n'); f.write_bytes(b'changed')
            with self.assertRaises(AssertionError): pub.cleanup_payload(root,'f.sft',r,index)
            self.assertTrue(f.exists())
            f.write_bytes(b'original'); self.assertEqual(pub.cleanup_payload(root,'f.sft',r,index),8)
            self.assertFalse(f.exists())
    def test_conflicting_remote_is_never_overwritten(self):
        client = FakeClient(); client.values['key'] = b'old'
        with self.assertRaises(AssertionError):
            pub.ensure_remote(client, 'key', b'new')
        self.assertEqual(client.values['key'], b'old')

    def test_readback_failure_is_not_success(self):
        with self.assertRaises(AssertionError):
            pub.ensure_remote(FakeClient(corrupt=True), 'key', b'content')

    def test_missing_payload_is_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(pub.verify_local(Path(folder), 'scene/file.sft', {'scene':'scene'}, 'streaming'))

    def test_receipt_failure_keeps_local_and_no_done_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'feature.sft'; path.write_bytes(b'feature')
            data = {'content':b'feature', 'sidecar':b'{}', 'index':b'index',
                    'index_relative':'prefix_index.npy', 'fingerprint':pub.fingerprint(path),
                    'identity':{'scene':'scene'}}
            with patch.object(pub, 'verify_local', return_value=data):
                with self.assertRaises(RuntimeError):
                    pub.publish_one(FakeClient(fail_receipt=True), root,'feature.sft',{},'streaming',root/'receipts')
            self.assertTrue(path.exists())
            self.assertFalse((root/'receipts').exists())

    def test_success_commits_receipt_but_does_not_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'feature.sft'; path.write_bytes(b'feature')
            data = {'content':b'feature', 'sidecar':b'{}', 'index':b'index',
                    'index_relative':'prefix_index.npy', 'fingerprint':pub.fingerprint(path),
                    'identity':{'scene':'scene'}}
            with patch.object(pub, 'verify_local', return_value=data):
                result, written = pub.publish_one(FakeClient(), root,'feature.sft',{},'streaming',root/'receipts')
            self.assertTrue(written)
            self.assertTrue(path.exists())
            self.assertEqual(result['status'],'REMOTE_VERIFIED')
            self.assertTrue(all(o['uri'].startswith('fj:s3://baiqiao/InsScene-15K/') for o in result['objects'].values()))
            self.assertEqual(pub.refresh_done_index(root/'receipts',root/'done.txt',{'feature.sft':{}}),1)
            self.assertEqual((root/'done.txt').read_text(), 'feature.sft\n')


if __name__=='__main__':
    unittest.main()
