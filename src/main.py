import os
import pickle
import cv2

from moviepy.editor import VideoFileClip

from config import cfg
from detect import detect, load_model
from train import get_features, train_model


class FrameProcessor:

    def __init__(self, svc, X_scaler):
        self.svc = svc
        self.X_scaler = X_scaler

    def run_on_frame(self, rgb_img):
        return detect(svc=self.svc, X_scaler=self.X_scaler, image=rgb_img)

def initialize_model():
    if not os.path.exists(cfg.MODEL_BIN):
        car_features, notcar_features = get_features()
        svc, X_scaler = train_model(car_features, notcar_features)
        model = {'svc': svc,
                 'X_scaler': X_scaler}

        # Save the model on disk
        with open(cfg.MODEL_BIN, 'wb') as f:
            pickle.dump(model, f)
    else:
        svc, X_scaler = load_model()

    return svc, X_scaler

def get_output_path(out_dir, img_name):
    img_name_no_ext = os.path.splitext(img_name)[0]
    return os.path.join(out_dir, img_name_no_ext + cfg.IMG_EXT)

def run_on_test_images(test_imgs_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    svc, X_scaler = initialize_model()
    frame_processor = FrameProcessor(svc, X_scaler)

    imgs = os.listdir(test_imgs_dir)
    print("Loaded test images {}".format(imgs))

    for img_path in imgs:
        img = cv2.imread(os.path.join(test_imgs_dir, img_path))
        detection = frame_processor.run_on_frame(img)
        out_fname = get_output_path(out_dir, img_path)
        cv2.imwrite(out_fname, detection)


def main(test_video, out_video):

    svc, X_scaler = initialize_model()
    frame_processor = FrameProcessor(svc, X_scaler)

    clip1 = VideoFileClip(test_video)
    white_clip = clip1.fl_image(frame_processor.run_on_frame)
    white_clip.write_videofile(out_video, audio=False)


if __name__ == "__main__":
    main(cfg.TEST_VIDEO, cfg.OUT_VIDEO)
    # run_on_test_images(cfg.TEST_DIR, cfg.OUT_DIR)
