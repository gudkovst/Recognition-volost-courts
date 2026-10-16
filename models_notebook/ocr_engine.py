import json
from pathlib import Path

import tensorflow as tf

from ocr_preprocessing import prepare_image


MODEL_NAMES = {
    "filled_long": "обучена на основных предварительно обработанных строках",
    "filled_short": "обучена на коротких (из бокового столбца) обработанных строках",
    "first20str1": "обучена на коротких и основных строках",
    "long_first_20str": "обучена на основных строках",
    "short_first_20str": "обучена на коротких строках",
}


class OCREngine:

    def __init__(
        self,
        model_name,
        models_dir="models",
        checkpoint=None
    ):

        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"Неизвестная модель: {model_name}"
            )

        self.model_name = model_name
        self.model_dir = Path(models_dir) / model_name

        config_path = self.model_dir / "best.ckpt.json"
        model_path = self.model_dir / "best.ckpt"

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:
            config = json.load(f)

        self.charset = config[
            "scenario"
        ]["data"]["codec"]["charset"]

        print("Загрузка модели:", model_name)

        self.model = tf.saved_model.load(
            str(model_path)
        )

        if checkpoint is not None:

            checkpoint = Path(checkpoint)

            if checkpoint.is_dir():
                checkpoint_path = (
                    tf.train.latest_checkpoint(
                        str(checkpoint)
                    )
                )
            else:
                checkpoint_path = str(checkpoint)

            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"Checkpoint не найден: {checkpoint}"
                )

            print(
                "Восстановление весов:",
                checkpoint_path
            )

            restore = tf.train.Checkpoint(
                model=self.model
            )

            status = restore.restore(
                checkpoint_path
            )

            status.assert_existing_objects_matched()
            status.expect_partial()

        self.predict = self.model.signatures[
            "serving_default"
        ]

        print("Модель готова.")

    def recognize(self, image_path):

        image = prepare_image(image_path)

        image = image[None, :, :, None]

        outputs = self.predict(
            img=tf.convert_to_tensor(
                image,
                dtype=tf.uint8
            ),
            img_len=tf.constant(
                [[image.shape[1]]],
                dtype=tf.int32
            )
        )

        indices = outputs[
            "root_2"
        ].numpy()[0]

        text = "".join(
            self.charset[int(index)]
            for index in indices
            if 0 <= int(index) < len(self.charset)
        )

        return text

    def recognize_folder(
        self,
        folder,
        extensions=(
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff"
        )
    ):

        folder = Path(folder)

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in extensions
        )

        results = []

        for image_path in files:

            try:
                text = self.recognize(
                    image_path
                )

                results.append({
                    "file": image_path.name,
                    "text": text,
                    "error": ""
                })

            except Exception as error:

                results.append({
                    "file": image_path.name,
                    "text": "",
                    "error": str(error)
                })

        return results


def list_models():
    return MODEL_NAMES.copy()