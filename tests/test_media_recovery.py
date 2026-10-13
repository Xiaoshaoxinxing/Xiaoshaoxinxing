"""Video fallback must preserve samples and sample actual decoded frames."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from mer_omni.media import decode_video, _decode_video_sequential


class VideoRecoveryTests(unittest.TestCase):
    def test_overstated_container_count_matches_actual_frame_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'thirteen.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25, (32, 24))
            self.assertTrue(writer.isOpened())
            for index in range(13):
                writer.write(np.full((24, 32, 3), index * 15, dtype=np.uint8))
            writer.release()
            expected = decode_video(path, 8, 32 * 24)
            original_capture = cv2.VideoCapture

            class OverstatedCapture:
                def __init__(self, filename):
                    self.capture = original_capture(filename)

                def get(self, property_id):
                    if property_id == cv2.CAP_PROP_FRAME_COUNT:
                        return 15
                    return self.capture.get(property_id)

                def __getattr__(self, name):
                    return getattr(self.capture, name)

            with patch('cv2.VideoCapture', OverstatedCapture), self.assertWarns(RuntimeWarning):
                actual = decode_video(path, 8, 32 * 24)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(len(np.unique(actual[:, 0, 0, 0])), 8)

    def test_decoder_errors_are_not_treated_as_short_valid_video(self):
        bad = subprocess.CompletedProcess([], 0, b'{"streams":[{"nb_read_frames":"13"}]}',
                                          b'corrupt frame')
        with patch('mer_omni.media.subprocess.run', return_value=bad):
            with self.assertRaisesRegex(ValueError, 'verification failed'):
                _decode_video_sequential('corrupt.avi', 8, 50176)


if __name__ == '__main__':
    unittest.main()
