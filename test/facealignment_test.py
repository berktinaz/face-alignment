import unittest
import face_alignment
import dlib


class Tester(unittest.TestCase):
    def test_predict_points(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False)
        fa.get_landmarks('assets/aflw-test.jpg')
    def test_predict_points_given_box_input(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False)
        fa.get_landmarks('assets/aflw-test.jpg', False, [dlib.drectangle( 135,187,329,377)] ) #coordinates obtained using SFD_pytorch with enlarged box

if __name__ == '__main__':
    unittest.main()
