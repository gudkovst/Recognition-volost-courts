import json
import unicodedata
from pathlib import Path

import numpy as np
import tensorflow as tf

from ocr_preprocessing import prepare_image


IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


def find_pairs(folder):
    """
    Находит пары вида:
        image.jpg
        image.gt.txt
    """

    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Папка не найдена: {folder}"
        )

    pairs = []

    for image_path in sorted(folder.iterdir()):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        text_path = image_path.with_suffix(".gt.txt")

        if text_path.is_file():
            pairs.append((image_path, text_path))
        else:
            print(
                "Пропущено изображение без разметки:",
                image_path.name
            )

    return pairs


class OCRTrainer:

    def __init__(
        self,
        model_name,
        models_dir="models",
        learning_rate=0.00001,
        experiment_name="experiment_001",
        results_dir="results/finetuned_models",
    ):

        self.model_name = model_name

        self.model_dir = (
            Path(models_dir) / model_name
        )

        self.experiment_dir = (
            Path(results_dir)
            / model_name
            / experiment_name
        )

        self.best_dir = (
            self.experiment_dir / "best"
        )

        self.latest_dir = (
            self.experiment_dir / "latest"
        )

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        config_path = (
            self.model_dir / "best.ckpt.json"
        )

        saved_model_path = (
            self.model_dir / "best.ckpt"
        )

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:
            config = json.load(f)

        self.charset = config[
            "scenario"
        ]["data"]["codec"]["charset"]

        self.char_to_index = {
            char: i
            for i, char in enumerate(self.charset)
        }

        print("Загрузка модели:", model_name)

        self.model = tf.saved_model.load(
            str(saved_model_path)
        )

        self.variables = list(
            self.model.trainable_variables
        )

        self.optimizer = (
            tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            )
        )

        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer
        )

        self.best_manager = (
            tf.train.CheckpointManager(
                self.checkpoint,
                directory=str(self.best_dir),
                max_to_keep=1
            )
        )

        self.latest_manager = (
            tf.train.CheckpointManager(
                self.checkpoint,
                directory=str(self.latest_dir),
                max_to_keep=2
            )
        )

        self.history_path = (
            self.experiment_dir / "history.json"
        )

        self.settings_path = (
            self.experiment_dir / "settings.json"
        )

        self.history = []

        self.best_val_loss = float("inf")

        print(
            "Обучаемых переменных:",
            len(self.variables)
        )

    # ========================================================
    # ЗАГРУЗКА ОДНОЙ ПАРЫ
    # ========================================================

    def load_sample(self, image_path, text_path):

        reference = Path(text_path).read_text(
            encoding="utf-8-sig"
        )

        reference = unicodedata.normalize(
            "NFC",
            reference.strip()
        )

        if not reference:
            raise ValueError(
                f"Пустой текст: {text_path}"
            )

        unknown = (
            set(reference)
            - set(self.char_to_index)
        )

        if unknown:
            raise ValueError(
                f"Неизвестные символы {unknown} "
                f"в файле {text_path}"
            )

        labels = [
            self.char_to_index[char]
            for char in reference
        ]

        if 0 in labels:
            raise ValueError(
                "В тексте обнаружен индекс CTC blank."
            )

        image = prepare_image(image_path)

        image = image[None, :, :, None]

        inputs = {
            "img": tf.convert_to_tensor(
                image,
                dtype=tf.uint8
            ),
            "img_len": tf.constant(
                [[image.shape[1]]],
                dtype=tf.int32
            )
        }

        label_indices = [
            [0, i]
            for i in range(len(labels))
        ]

        sparse_labels = tf.SparseTensor(
            indices=tf.constant(
                label_indices,
                dtype=tf.int64
            ),
            values=tf.constant(
                labels,
                dtype=tf.int32
            ),
            dense_shape=tf.constant(
                [1, len(labels)],
                dtype=tf.int64
            )
        )

        sparse_labels = tf.sparse.reorder(
            sparse_labels
        )

        return inputs, sparse_labels, labels

    # ========================================================
    # ПРОВЕРКА ДЛИНЫ CTC
    # ========================================================

    def check_ctc_length(
        self,
        inputs,
        labels,
        image_path
    ):

        outputs = self.model(
            inputs,
            training=False
        )

        output_length = int(
            outputs["out_len"][0]
        )

        required_length = (
            len(labels)
            + sum(
                labels[i] == labels[i - 1]
                for i in range(1, len(labels))
            )
        )

        if required_length > output_length:

            raise ValueError(
                f"Слишком длинный текст для CTC: "
                f"{image_path}. "
                f"Нужно {required_length}, "
                f"доступно {output_length}."
            )

    # ========================================================
    # CTC LOSS
    # ========================================================

    def calculate_loss(self, outputs, labels):

        losses = tf.nn.ctc_loss(
            labels=labels,
            logits=outputs["logits"],
            label_length=None,
            logit_length=outputs["out_len"],
            logits_time_major=False,
            blank_index=0
        )

        loss = tf.reduce_mean(losses)

        if not bool(tf.math.is_finite(loss)):
            raise RuntimeError(
                "CTC loss содержит NaN или Inf."
            )

        return loss

    # ========================================================
    # ОДИН ШАГ ОБУЧЕНИЯ
    # ========================================================

    def train_step(self, inputs, labels):

        with tf.GradientTape() as tape:

            outputs = self.model(
                inputs,
                training=True
            )

            loss = self.calculate_loss(
                outputs,
                labels
            )

        gradients = tape.gradient(
            loss,
            self.variables
        )

        for variable, gradient in zip(
            self.variables,
            gradients
        ):

            if gradient is None:
                raise RuntimeError(
                    f"Нет градиента: {variable.name}"
                )

            if not bool(
                tf.reduce_all(
                    tf.math.is_finite(gradient)
                )
            ):
                raise RuntimeError(
                    f"Некорректный градиент: "
                    f"{variable.name}"
                )

        self.optimizer.apply_gradients(
            zip(gradients, self.variables)
        )

        return float(loss)

    # ========================================================
    # ОЦЕНКА
    # ========================================================

    def evaluate(self, pairs):

        losses = []

        for image_path, text_path in pairs:

            inputs, sparse_labels, labels = (
                self.load_sample(
                    image_path,
                    text_path
                )
            )

            self.check_ctc_length(
                inputs,
                labels,
                image_path
            )

            outputs = self.model(
                inputs,
                training=False
            )

            loss = self.calculate_loss(
                outputs,
                sparse_labels
            )

            losses.append(float(loss))

        return (
            float(np.mean(losses))
            if losses
            else None
        )

    # ========================================================
    # ВОССТАНОВЛЕНИЕ
    # ========================================================

    def restore_latest(self):

        checkpoint_path = (
            self.latest_manager.latest_checkpoint
        )

        if checkpoint_path is None:
            raise FileNotFoundError(
                "Не найдена контрольная точка для "
                f"продолжения обучения: {self.latest_dir}"
            )

        print(
            "Восстановление:",
            checkpoint_path
        )

        self.checkpoint.restore(
            checkpoint_path
        ).assert_existing_objects_matched()

        if self.history_path.exists():

            with open(
                self.history_path,
                "r",
                encoding="utf-8"
            ) as f:
                self.history = json.load(f)

        if self.history:

            completed_epochs = (
                self.history[-1]["epoch"]
            )

            validation_losses = [
                record["validation_loss"]
                for record in self.history
                if record["validation_loss"] is not None
            ]

            if validation_losses:
                self.best_val_loss = min(
                    validation_losses
                )

            print(
                "Уже выполнено эпох:",
                completed_epochs
            )

            return completed_epochs

        return 0

    # ========================================================
    # СОХРАНЕНИЕ ИСТОРИИ
    # ========================================================

    def save_history(self):

        with open(
            self.history_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.history,
                f,
                ensure_ascii=False,
                indent=2
            )

    # ========================================================
    # ОБУЧЕНИЕ
    # ========================================================

    def train(
        self,
        train_dir,
        validation_dir=None,
        epochs=5,
        resume=False,
    ):

        train_pairs = find_pairs(train_dir)

        if not train_pairs:
            raise ValueError(
                "Нет обучающих пар."
            )

        validation_pairs = []

        if validation_dir is not None:

            validation_pairs = find_pairs(
                validation_dir
            )

        print(
            "\nОбучающих пар:",
            len(train_pairs)
        )

        print(
            "Проверочных пар:",
            len(validation_pairs)
        )

        start_epoch = 0

        if resume:
            start_epoch = self.restore_latest()

        settings = {
            "model_name": self.model_name,
            "train_dir": str(train_dir),
            "validation_dir": (
                str(validation_dir)
                if validation_dir
                else None
            ),
            "epochs": epochs,
            "learning_rate": float(
                self.optimizer.learning_rate.numpy()
            )
        }

        with open(
            self.settings_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                settings,
                f,
                ensure_ascii=False,
                indent=2
            )

        for epoch in range(
            start_epoch + 1,
            start_epoch + epochs + 1
        ):

            print(
                f"\nЭпоха {epoch}"
            )

            np.random.shuffle(
                train_pairs
            )

            losses = []

            for index, (
                image_path,
                text_path
            ) in enumerate(
                train_pairs,
                start=1
            ):

                inputs, sparse_labels, labels = (
                    self.load_sample(
                        image_path,
                        text_path
                    )
                )

                self.check_ctc_length(
                    inputs,
                    labels,
                    image_path
                )

                loss = self.train_step(
                    inputs,
                    sparse_labels
                )

                losses.append(loss)

                print(
                    f"{index}/{len(train_pairs)} "
                    f"{image_path.name}: "
                    f"loss={loss:.4f}"
                )

            train_loss = float(
                np.mean(losses)
            )

            val_loss = None

            if validation_pairs:

                val_loss = self.evaluate(
                    validation_pairs
                )

            print(
                "Train loss:",
                train_loss
            )

            print(
                "Validation loss:",
                val_loss
            )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss
            }

            self.history.append(
                record
            )

            # Сохраняем последнее состояние.

            latest_path = (
                self.latest_manager.save()
            )

            print(
                "Latest checkpoint:",
                latest_path
            )

            # Сохраняем лучший результат.

            if val_loss is not None:

                if val_loss < self.best_val_loss:

                    self.best_val_loss = val_loss

                    best_path = (
                        self.best_manager.save()
                    )

                    print(
                        "Новая лучшая модель:",
                        best_path
                    )

            else:

                best_path = (
                    self.best_manager.save()
                )

                print(
                    "Модель сохранена:",
                    best_path
                )

            self.save_history()

        print("\nОбучение завершено.")

        return self.history