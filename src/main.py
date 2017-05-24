import os
import pickle

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






def main(test_video, out_video):

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

    frame_processor = FrameProcessor(svc, X_scaler)

    clip1 = VideoFileClip(test_video)
    white_clip = clip1.fl_image(frame_processor.run_on_frame)
    white_clip.write_videofile(out_video, audio=False)


if __name__ == "__main__":
    main(cfg.TEST_VIDEO, cfg.OUT_VIDEO)
