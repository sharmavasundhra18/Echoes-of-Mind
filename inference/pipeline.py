from inference.audio_inference import AudioInference
from inference.text_inference import TextInference
from models.fusion import MultimodalFusion


class EchoesPipeline:

    def __init__(self):

        self.audio_model = AudioInference()

        self.text_model = TextInference()

        self.fusion = MultimodalFusion()

    def analyze(self, audio_path, text):

        # audio prediction
        audio_emotion, audio_conf = self.audio_model.predict(audio_path)

        # text prediction
        text_sentiment, text_conf = self.text_model.predict(text)

        # fusion
        fusion_result = self.fusion.fuse(
            audio_emotion,
            audio_conf,
            text_sentiment,
            text_conf
        )

        return {
            "audio_emotion": audio_emotion,
            "audio_confidence": audio_conf,
            "text_sentiment": text_sentiment,
            "text_confidence": text_conf,
            "fusion_result": fusion_result
        }