import json
import random
import unicodedata
from pathlib import Path
from string import ascii_letters
from typing import Any

from confusable_homoglyphs import confusables
from nltk.corpus import stopwords

from .._manipulator import Manipulator
from . import _prompts
from ._perturb_candidate import PerturbCandidateList

BASE = Path(__file__).parent / "files"


class TextualPerturbationManipulator(Manipulator):
    """Manipulator that handles Text Pertubations."""

    def __init__(
        self,
        homophone_file: str | Path = Path(BASE / "homophone_mapping.json"),
        synonym_file: str | Path = Path(BASE / "synonym_mapping.json"),
    ) -> None:
        """
        Initialize the TextualPerturbationManipulator.

        :param homophone_file: The json file path that contains homophone mappings.
        :param synonym_file: The json file path that contains synonym mappings.
        """
        self.homoglyphs = self._get_latin_homoglyph_dict()

        self.invisible_chars = list(
            set(
                [
                    ch
                    for codepoint in range(0x110000)
                    if unicodedata.category(ch := chr(codepoint)) in {"Cf", "Mn"}
                ]
            )
        )
        self.stop_words = list(set(stopwords.words("english")))

        with open(synonym_file, "r", encoding="utf-8") as f:
            self.synonym_map = json.load(f)

        with open(homophone_file, "r", encoding="utf-8") as f:
            self.homophone_map = json.load(f)

        self.adversarial_suffixes = _prompts.ADVERSARIAL_SUFFIXES
        self.context_distractors = _prompts.CONTEXT_DISTRACTORS
        self.reinforcement_phrases = _prompts.REINFORCEMENT_PHRASES

        self.obj_pertubations = [
            self.fragmentation,
            self.character_noise,
            self.homophone_substitution,
            self.synonym_substitution,
            self.ata_saliency,
        ]
        self.prompt_pertubations = [
            self.universal_suffix_injection,
            self.context_rot_injection,
            self.task_reinforcement,
        ]

    def manipulate(self, candidates: PerturbCandidateList, **kwargs: Any) -> list[str]:
        """
        The manipulation function for the Manipulator.

        :param candidates: The candidates to manipulate.
        :param kwargs: Keyword arguments to pass to the manipulation function.
        :returns: The result of the manipulation.
        """
        for candidate in candidates:
            assert len(candidate.text_perturbation) == (
                len(self.obj_pertubations) + len(self.prompt_pertubations)
            ), f"ERROR: Genome has length {len(candidate.text_perturbation)}, but {(len(self.obj_pertubations) + len(self.prompt_pertubations))} text pertubations found."
            for scale, pert in zip(
                candidate.text_perturbation[: len(self.obj_pertubations)], self.obj_pertubations
            ):
                if scale <= 0.01:
                    continue
                candidate.objects_str = pert(candidate.objects_str, scale)
            for scale, pert in zip(
                candidate.text_perturbation[len(self.obj_pertubations) :], self.prompt_pertubations
            ):
                candidate.prompt_str = pert(candidate.prompt_str, scale)

        return [c.format_prompt() for c in candidates]

    # --------------- Pertubations
    def fragmentation(self, text: str, scale: float = 0.0) -> str:
        """Randomly split words at an interior position to simulate OCR-like fragmentation.

        :param text: Input text string.
        :param scale: Severity in [0.0, 1.0]; controls splitting probability.
        :returns: Text with words randomly split by a space.
        """
        if not text or len(text) < 2:
            return text
        probability = self._scale_to_prob(scale)
        result = []
        for word in text.split():
            if len(word) > 3 and random.random() < probability:
                split_idx = random.randint(1, len(word) - 1)
                result.append(word[:split_idx] + " " + word[split_idx:])
            else:
                result.append(word)
        return " ".join(result)

    def character_noise(self, text: str, scale: float = 0.0) -> str:
        """Substitute visually similar Unicode homoglyphs and insert zero-width characters.

        :param text: Input text string.
        :param scale: Severity in [0.0, 1.0]; controls substitution and insertion probability.
        :returns: Text with homoglyph substitutions and invisible Unicode characters injected.
        """
        if not text:
            return text
        probability = self._scale_to_prob(scale)
        result = ""
        for char in text:
            current_char = char
            if char in self.homoglyphs and random.random() < probability:
                current_char = random.choice(self.homoglyphs[char])
            result += current_char
            if char.isalnum() and random.random() < probability:
                result += random.choice(self.invisible_chars)
        return result

    def ata_saliency(self, text: str, scale: float = 0.0) -> str:
        """Introduce typos into salient (non-stop) words at a scale-dependent rate.

        :param text: Input text string.
        :param scale: Severity in [0.0, 1.0]; controls the fraction of salient words corrupted.
        :returns: Text with typos injected into salient content words.
        """
        perturbation_rate = self._scale_to_prob(scale)
        words = text.split()
        perturbed_words = []
        for word in words:
            clean = re.sub(r"[^\w]", "", word).lower()
            is_salient = (clean not in self.stop_words) and (len(clean) > 2)

            if is_salient and random.random() < perturbation_rate:
                typo = self._generate_typo(clean)
                if word[0].isupper():
                    typo = typo.capitalize()
                if not word[-1].isalnum():
                    typo += word[-1]
                perturbed_words.append(typo)
            else:
                perturbed_words.append(word)
        return " ".join(perturbed_words)

    def homophone_substitution(self, text: str, scale: float = 0.0) -> str:
        """Replace comma-separated object labels with homophones from the loaded mapping.

        :param text: Comma-separated object label string.
        :param scale: Severity in [0.0, 1.0]; controls substitution probability and option range.
        :returns: Label string with homophones substituted.
        """
        if not text:
            return text
        objects = [obj.strip() for obj in text.split(",")]
        transformed_objects = []

        probability = self._scale_to_prob(scale)
        max_options_idx = 1 + int(scale * 4)

        for obj in objects:
            if obj in self.homophone_map and random.random() < probability:
                entry = self.homophone_map[obj]
                homophone_list = [entry] if isinstance(entry, str) else entry

                candidates = homophone_list[:max_options_idx]
                if candidates:
                    transformed_objects.append(random.choice(candidates))
                else:
                    transformed_objects.append(obj)
            else:
                transformed_objects.append(obj)
        return ", ".join(transformed_objects)

    def synonym_substitution(self, text: str, scale: float = 0.0) -> str:
        """Replace labels with synonyms from the loaded mapping.

        :param text: object label string.
        :param scale: Severity in [0.0, 1.0]; controls substitution probability and option range.
        :returns: Label string with synonyms substituted.
        """
        if not text:
            return text
        objects = [obj.strip() for obj in text.split(",")]
        transformed = []

        probability = self._scale_to_prob(scale)
        max_options_idx = 1 + int(scale * 4)

        for obj in objects:
            if obj in self.synonym_map and random.random() < probability:
                entry = self.synonym_map[obj]
                opts = [entry] if isinstance(entry, str) else entry

                candidates = opts[:max_options_idx]
                if candidates:
                    transformed.append(random.choice(candidates))
                else:
                    transformed.append(obj)
            else:
                transformed.append(obj)
        return ", ".join(transformed)

    def universal_suffix_injection(self, prompt: str, scale: float = 0.0) -> str:
        """Append one or more adversarial suffix strings to the prompt.

        :param prompt: Original prompt string.
        :param scale: Severity in [0.0, 1.0]; higher values append more suffixes.
        :returns: Prompt with adversarial suffixes appended.
        """
        count = 1 + int(scale * 4)
        suffixes = [random.choice(self.adversarial_suffixes) for _ in range(count)]
        return f"{prompt} {' '.join(suffixes)}"

    def context_rot_injection(self, prompt: str, scale: float = 0.0) -> str:
        """Prepend a randomly corrupted distractor sentence to the prompt.

        :param prompt: Original prompt string.
        :param scale: Severity in [0.0, 1.0]; controls the heavy-typo corruption rate of the distractor.
        :returns: Prompt prefixed with a corrupted distractor sentence.
        """
        clean = random.choice(self.context_distractors)
        rate = 0.1 + (max(0.0, min(scale, 1.0)) * 0.8)
        rotted = self._apply_heavy_typos(clean, rate=rate)
        return f"{rotted}... Now, {prompt}"

    def task_reinforcement(self, prompt: str, scale: float = 0.0) -> str:
        """Prepend an escalating adversarial reinforcement phrase to the prompt.

        :param prompt: Original prompt string.
        :param scale: Severity in [0.0, 1.0]; selects a more alarming phrase at higher values.
        :returns: Prompt prefixed with a reinforcement phrase.
        """
        max_idx = len(self.reinforcement_phrases) - 1
        idx = min(int(scale * max_idx), max_idx)
        return f"{self.reinforcement_phrases[idx]} {prompt}"

    # --------------- Helpers

    def _scale_to_prob(self, scale: float) -> float:
        """Maps float scale 0.0-1.0 to a probability 0.1-0.9.

        :param scale: Severity in [0.0, 1.0].
        :returns: Probability in [0.1, 0.9].
        """
        scale = max(0.0, min(scale, 1.0))
        return 0.1 + (scale * 0.8)

    def _generate_typo(self, word: str) -> str:
        """Return a single-character or swap typo for ``word``.

        :param word: Source word (returned unchanged if shorter than 3 characters).
        :returns: Typo-corrupted word string.
        """
        if len(word) < 3:
            return word
        word_list = list(word)
        if random.random() < 0.5:  # Swap
            idx = random.randint(0, len(word) - 2)
            word_list[idx], word_list[idx + 1] = word_list[idx + 1], word_list[idx]
        else:  # Replace
            idx = random.randint(1, len(word) - 1)
            word_list[idx] = chr(random.randint(97, 122))
        return "".join(word_list)

    def _apply_heavy_typos(self, text: str, rate: float = 0.3) -> str:
        """Corrupt a string by randomly swapping, replacing, deleting, or duplicating characters.

        :param text: Input string.
        :param rate: Fraction of characters to corrupt (0.0–1.0).
        :returns: Corrupted string.
        """
        chars = list(text)
        length = len(chars)
        num_changes = int(length * rate)
        for _ in range(num_changes):
            if length < 2:
                break
            op = random.randint(0, 4)
            idx = random.randint(0, length - 2)
            if op < 1:  # Swap
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif 1 <= op < 2:  # Replace
                chars[idx] = random.choice(string.ascii_letters)
            elif 2 <= op < 3 and len(chars) > 5:  # Delete
                chars.pop(idx)
                length -= 1
            elif 3 <= op < 4:  #  Insert
                chars.insert(idx, chars[idx])
                length += 1
        return "".join(chars)

    @staticmethod
    def _get_latin_homoglyph_dict() -> dict[str, list[str]]:
        """
        Generate a dictionary of homoglyphs for all latin characters.

        :returns: Dictionary of homoglyphs.
        """
        out = {}

        for ch in ascii_letters:
            matches = confusables.is_confusable(ch, greedy=True)
            glyphs = set()

            if matches:
                for match in matches:
                    for item in match["homoglyphs"]:
                        glyphs.add(item["c"])

            glyphs.discard(ch)
            if len(glyphs) > 0:
                out[ch] = list(sorted(glyphs))
        return out

    def synthesize(self, z: Any) -> Any:
        raise NotImplementedError("Synthesize not implemented.")
