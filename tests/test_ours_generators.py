"""v20.2: the generator sweep (OURS_VERSIONS "v20.2").

Ours with each of seven pinned generators as its one LLM, reader and tree summariser alike,
each building its own trees in ours' own cache; the frozen reader's path as it was. The
harness renders every chat prompt with one fixed date, so a template that would print
today's date (Llama 3.2's) prints the date Llama 3.1's template prints; and the GPU cap can
be raised for a model the L4's 20 GiB does not hold."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ours import config as settings
from ours import model as om
from scripts import dev_ours
from src.retrievers.m4_raptor import M4_SUBSTRATE_NAMESPACE, RaptorSystem
from tests.test_ours_smoke import (
    ANSWERS,
    DECOMPOSITIONS,
    FakeReader,
    TwoHotpotUnits,
    fake_generate_batch,
    hotpot_unit,
    llama_config,
    ours_fakes,
)

SWEEP = ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
         "Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
         "microsoft/Phi-4-mini-instruct", "Qwen/Qwen2.5-7B-Instruct",
         "Qwen/Qwen2.5-14B-Instruct"]
QWEN3B = "Qwen/Qwen2.5-3B-Instruct"
PHI = "microsoft/Phi-4-mini-instruct"


def short_units():
    """Two hotpot units, each with a document under M4's filter."""
    from tests.test_ours_short_documents import unit_with_short_document

    return [unit_with_short_document(0), unit_with_short_document(1)]


class _ShortUnits(TwoHotpotUnits):
    def __init__(self) -> None:
        super().__init__()
        self.units = short_units()


class _MixedUnits(TwoHotpotUnits):
    """One unit M4's filter leaves whole, one it would cut a document from."""

    def __init__(self) -> None:
        super().__init__()
        self.units = [hotpot_unit(0), short_units()[1]]


def tree_dirs(cache: Path, namespace: str) -> list[Path]:
    return sorted(p.parent for p in (cache / namespace).rglob("manifest.json"))


class TestTheSweep(unittest.TestCase):
    """Seven generators, smallest first, each at the revision the hub served on 2026-11-19."""

    def test_seven_generators_smallest_first_none_of_them_the_reader(self):
        self.assertEqual(list(settings.SWEEP_GENERATORS), SWEEP)
        self.assertNotIn(settings.READER, settings.SWEEP_GENERATORS)
        for revision in settings.SWEEP_GENERATORS.values():
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_the_sweep_runs_narrativeqa_then_multihop(self):
        self.assertEqual(dev_ours.SWEEP_BENCHMARKS, ("narrativeqa", "multihop_rag"))

    def test_each_generator_has_its_own_bank_directory(self):
        names = [dev_ours.sweep_bank(Path("x"), g).name for g in SWEEP]
        self.assertEqual(names[0], "Llama-3.2-1B-Instruct")
        self.assertEqual(len(set(names)), len(SWEEP))


class TestTheGuard(unittest.TestCase):
    """One model plays every LLM role: the reader or a sweep generator, never two models."""

    def test_every_sweep_generator_is_accepted_as_reader_and_summariser(self):
        for generator in SWEEP:
            config = dev_ours.generator_config(generator)
            self.assertEqual(config.generation.model, generator)
            self.assertEqual(config.m4.summary_model, generator)
            om.OursSystem(config=config)

    def test_the_readers_config_is_the_llama_config(self):
        self.assertEqual(dev_ours.generator_config(settings.READER), dev_ours.llama_config())
        self.assertEqual(dev_ours.llama_config(), llama_config())

    def test_a_model_outside_the_sweep_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "OURS REFUSED: reader 'Qwen/Qwen2.5-0.5B-Instruct'"):
            om.OursSystem(config=dev_ours.generator_config("Qwen/Qwen2.5-0.5B-Instruct"))

    def test_a_reader_with_another_summariser_is_refused(self):
        from dataclasses import replace

        for reader, summariser in ((QWEN3B, settings.READER), (settings.READER, QWEN3B), (QWEN3B, PHI)):
            config = dev_ours.generator_config(reader)
            config = replace(config, m4=replace(config.m4, summary_model=summariser))
            with self.assertRaisesRegex(SystemExit, "tree summariser"):
                om.OursSystem(config=config)


class TestOwnTrees(unittest.TestCase):
    """A sweep generator's trees: M4's builder and key, in ours' own cache, one per generator.
    The frozen reader reads M4's banked trees as before."""

    def _index(self, cache, generator, units, reader=None, batches=None):
        def recording_batch(system_prompts, user_prompts, cfg=None, **kw):
            if batches is not None:
                batches.append(cfg.model)
            return fake_generate_batch(system_prompts, user_prompts, cfg, **kw)

        hits = []
        with ours_fakes(cache, reader or FakeReader(DECOMPOSITIONS, ANSWERS)), \
                mock.patch("src.models.generate_batch", recording_batch):
            system = om.OursSystem(config=dev_ours.generator_config(generator))
            for unit in units:
                system.index_items(unit.corpus)
                hits.append(system.tree_cache_hit)
        return system, hits

    def test_a_sweep_generators_tree_is_built_in_ours_cache_not_m4s(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            batches = []
            system, hits = self._index(cache, QWEN3B, [hotpot_unit(0)], batches=batches)
            self.assertIsInstance(system._m4, om.SweepTree)
            self.assertEqual(hits, [False])
            self.assertFalse((cache / M4_SUBSTRATE_NAMESPACE).exists())
            (built,) = tree_dirs(cache, settings.SWEEP_TREE_NAMESPACE)
            manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["extra"]["m4"]["summary_model"], QWEN3B)
            # The summaries were written by the generator itself.
            self.assertTrue(batches)
            self.assertEqual(set(batches), {QWEN3B})
            _, hits = self._index(cache, QWEN3B, [hotpot_unit(0)])
            self.assertEqual(hits, [True])

    def test_the_key_is_m4s_own(self):
        """The same builder, settings and key M4 would use under this generator."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            self._index(cache, QWEN3B, [hotpot_unit(0)])
            (built,) = tree_dirs(cache, settings.SWEEP_TREE_NAMESPACE)
            with ours_fakes(cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
                m4 = RaptorSystem(dev_ours.generator_config(QWEN3B))
                m4.topology_env_override = settings.TREE_BUILD_ENV
                warm_in_m4s_place = m4.substrate_warm_path(hotpot_unit(0).corpus)
                own = om.SweepTree(dev_ours.generator_config(QWEN3B))
                own.topology_env_override = settings.TREE_BUILD_ENV
                warm_in_ours = own.substrate_warm_path(hotpot_unit(0).corpus)
            self.assertIsNone(warm_in_m4s_place)
            self.assertEqual(Path(warm_in_ours), built)
            self.assertEqual(built.name, Path(warm_in_ours).name)

    def test_each_generator_gets_its_own_tree(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            self._index(cache, QWEN3B, [hotpot_unit(0)])
            _, hits = self._index(cache, PHI, [hotpot_unit(0)])
            self.assertEqual(hits, [False])
            built = tree_dirs(cache, settings.SWEEP_TREE_NAMESPACE)
            self.assertEqual(len(built), 2)
            models = {json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                      ["extra"]["m4"]["summary_model"] for d in built}
            self.assertEqual(models, {QWEN3B, PHI})

    def test_a_unit_with_a_short_document_takes_ours_whole_corpus_tree(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            system, _ = self._index(cache, QWEN3B, short_units()[:1])
            self.assertIsInstance(system._tree_system, om.WholeCorpusTree)
            self.assertEqual(len(tree_dirs(cache, settings.TREE_NAMESPACE)), 1)
            self.assertFalse((cache / settings.SWEEP_TREE_NAMESPACE).exists())
            self.assertFalse((cache / M4_SUBSTRATE_NAMESPACE).exists())

    def test_the_reader_reads_m4s_tree_as_before(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            system, _ = self._index(cache, settings.READER, [hotpot_unit(0)])
            self.assertIs(type(system._m4), RaptorSystem)
            self.assertEqual(len(tree_dirs(cache, M4_SUBSTRATE_NAMESPACE)), 1)
            self.assertFalse((cache / settings.SWEEP_TREE_NAMESPACE).exists())

    def test_every_call_of_a_row_is_the_generators(self):
        reader = FakeReader(DECOMPOSITIONS, ANSWERS)
        with tempfile.TemporaryDirectory() as td:
            system, _ = self._index(Path(td), PHI, [hotpot_unit(0)], reader=reader)
            with ours_fakes(td, reader):
                system.answer(hotpot_unit(0).queries[0].question_text)
        self.assertTrue(reader.cfgs)
        self.assertEqual({c.model for c in reader.cfgs}, {PHI})
        self.assertEqual({c.temperature for c in reader.cfgs}, {0.0})


class TestRevisions(unittest.TestCase):
    """The hub must still serve the run's own generator at its pin."""

    def _served(self, shas):
        api = mock.Mock()
        api.model_info.side_effect = lambda name: SimpleNamespace(sha=shas[name])
        return mock.patch("huggingface_hub.HfApi", return_value=api), api

    def _good(self, generator):
        revision = (settings.READER_REVISION if generator == settings.READER
                    else settings.SWEEP_GENERATORS[generator])
        return {generator: revision, settings.EMBEDDER: settings.EMBEDDER_REVISION,
                settings.SECOND_EMBEDDER: settings.SECOND_EMBEDDER_REVISION}

    def test_the_runs_generator_is_pinned_and_the_reader_not_asked(self):
        for generator in SWEEP:
            patch, api = self._served(self._good(generator))
            with patch:
                om.check_model_revisions(generator)
            asked = [c.args[0] for c in api.model_info.call_args_list]
            self.assertIn(generator, asked)
            self.assertNotIn(settings.READER, asked)
            moved = dict(self._good(generator), **{generator: "0" * 40})
            patch, _ = self._served(moved)
            with patch, self.assertRaisesRegex(SystemExit, "OURS REFUSED: the hub serves"):
                om.check_model_revisions(generator)

    def test_without_a_generator_it_pins_the_reader(self):
        patch, api = self._served(self._good(settings.READER))
        with patch:
            om.check_model_revisions()
        self.assertIn(mock.call(settings.READER), api.model_info.call_args_list)

    def test_a_cell_checks_its_own_generator(self):
        seen = []
        with tempfile.TemporaryDirectory() as td, ours_fakes(td, FakeReader(DECOMPOSITIONS, ANSWERS)), \
                mock.patch("ours.model.check_model_revisions", lambda *a: seen.append(a)):
            om.OursSystem(config=dev_ours.generator_config(PHI)).index_items(hotpot_unit(0).corpus)
        self.assertEqual(seen, [(PHI,)])


# --- the harness: one date in every chat prompt; the GPU cap -------------------------


class _Encoded(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """Records apply_chat_template's keyword arguments; renders a Llama-3.2-like date block."""

    eos_token, pad_token, pad_token_id, eos_token_id = "</s>", None, 0, 0
    padding_side = "right"

    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.template_kwargs.append(kwargs)
        date = kwargs.get("date_string", "TODAY")
        return f"Today Date: {date}|" + "|".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        texts = [text] if isinstance(text, str) else text
        ids = np.ones((len(texts), 3), dtype=np.int64)
        enc = _Encoded(input_ids=ids, attention_mask=ids)
        return enc if return_tensors else {"input_ids": list(ids[0])}

    def decode(self, row, skip_special_tokens=True):
        return "reply"


class FakeModel:
    device = "cpu"

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        return np.concatenate([input_ids, np.full((input_ids.shape[0], 2), 7)], axis=1)

    def named_parameters(self):
        return iter(())


def fake_torch():
    torch = types.ModuleType("torch")
    torch.no_grad = torch.inference_mode = contextlib.nullcontext
    torch.cuda = SimpleNamespace(is_available=lambda: False, synchronize=lambda: None)
    return torch


class TestOneDateInEveryPrompt(unittest.TestCase):
    """Llama 3.2's template prints today's date unless it is given one; Llama 3.1's prints
    26 Jul 2024 unless it is given one. Both are given 26 Jul 2024, so a cell and its trees
    read the same prompts on any day, and Llama 3.1's prompts are the ones it always read."""

    def test_the_date_is_llama_3_1s_own(self):
        from src.config import CHAT_TEMPLATE_DATE

        self.assertEqual(CHAT_TEMPLATE_DATE, "26 Jul 2024")

    def test_the_helper_passes_it(self):
        from src import models
        from src.config import CHAT_TEMPLATE_DATE

        tok = FakeTokenizer()
        text = models.chat_text(tok, "sys", "user")
        self.assertEqual(tok.template_kwargs, [{"date_string": CHAT_TEMPLATE_DATE}])
        self.assertEqual(text, f"Today Date: {CHAT_TEMPLATE_DATE}|sys|user")

    def test_the_reader_and_the_tree_summaries_both_render_through_it(self):
        from src import models
        from src.config import CHAT_TEMPLATE_DATE, GenerationConfig

        tok, cfg = FakeTokenizer(), GenerationConfig(model="org/some-local-model")
        self.addCleanup(models.reset_generate_calls)
        with mock.patch.dict(sys.modules, {"torch": fake_torch()}), \
                mock.patch.object(models, "load_generator", return_value=(tok, FakeModel())), \
                mock.patch.object(models, "configure_cuda_allocator", lambda: None):
            self.assertEqual(models.generate("sys", "user", cfg), "reply")
            self.assertEqual(models.generate_batch(["s1", "s2"], ["u1", "u2"], cfg), ["reply", "reply"])
        self.assertEqual(tok.template_kwargs, [{"date_string": CHAT_TEMPLATE_DATE}] * 3)


class TestGpuCap(unittest.TestCase):
    """20 GiB, as the matrix ran; THESIS_GENERATOR_MAX_MEMORY raises it for a larger GPU."""

    def test_the_default_is_the_matrixs(self):
        from src import models
        from src.config import GENERATOR_MAX_MEMORY

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("THESIS_GENERATOR_MAX_MEMORY", None)
            self.assertEqual(models.generator_max_memory(), {0: "20GiB", "cpu": "0GiB"})
        self.assertEqual(GENERATOR_MAX_MEMORY, {0: "20GiB", "cpu": "0GiB"})

    def test_the_environment_raises_it_and_nothing_spills_to_the_cpu(self):
        from src import models

        with mock.patch.dict(os.environ, {"THESIS_GENERATOR_MAX_MEMORY": "36GiB"}):
            self.assertEqual(models.generator_max_memory(), {0: "36GiB", "cpu": "0GiB"})

    def test_a_malformed_cap_is_refused(self):
        from src import models

        for bad in ("36", "36GB", "lots", "0GiB"):
            with mock.patch.dict(os.environ, {"THESIS_GENERATOR_MAX_MEMORY": bad}):
                with self.assertRaisesRegex(ValueError, "THESIS_GENERATOR_MAX_MEMORY"):
                    models.generator_max_memory()


# --- dev_ours: the sweep's cells and trees -----------------------------------------


class TestSweepCell(unittest.TestCase):
    """A sweep cell: the runner's own command with the generator, into its own bank under the
    sweep's directory; NarrativeQA, then MultiHop; the model is v20's."""

    def setUp(self):
        from ours.model import CODE_FINGERPRINT

        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        self.sweep, self.local = root / "drive" / "ours_v99_generators", root / "local"
        self.code = CODE_FINGERPRINT
        self.patches = [mock.patch.dict("scripts.report_generators.SWEEP_CODE",
                                        {"ours_v99_generators": "f" * 16}),
                        mock.patch.dict("scripts.report_ours.SAME_OUTPUTS", {self.code: "f" * 16})]
        for p in self.patches:
            p.start()
        self.commands = []

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.td.cleanup()

    def _runner(self, returncode=0):
        def run(command, cwd):
            self.commands.append(command)
            out = Path(command[command.index("--output") + 1])
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"query_id": "q0", "metadata": {"ours_code": self.code}}) + "\n")
            if not returncode:
                out.with_suffix(".summary.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=returncode)
        return run

    def _cell(self, benchmark="narrativeqa", generator=QWEN3B, local=True):
        with contextlib.redirect_stdout(io.StringIO()):
            dev_ours.scored_cell(benchmark, self.sweep, self.local if local else None,
                                 run=self._runner(), generator=generator)

    def test_the_runners_command_with_the_generator_into_its_bank(self):
        self._cell()
        command = self.commands[0]
        self.assertEqual(command[command.index("--generator") + 1], QWEN3B)
        self.assertEqual(command[command.index("--system") + 1], "OURS")
        self.assertEqual(command[-1], "--resume")
        self.assertEqual(Path(command[command.index("--output") + 1]),
                         self.local / "Qwen2.5-3B-Instruct" / "narrativeqa_OURS_validation.jsonl")
        bank = self.sweep / "Qwen2.5-3B-Instruct"
        self.assertEqual(sorted(p.name for p in bank.iterdir()),
                         ["narrativeqa_OURS_validation.jsonl", "narrativeqa_OURS_validation.summary.json"])

    def test_multihop_after_narrativeqa_for_each_generator(self):
        with self.assertRaisesRegex(SystemExit, "MultiHop runs last"):
            self._cell("multihop_rag")
        self._cell("narrativeqa")
        self._cell("multihop_rag")
        with self.assertRaisesRegex(SystemExit, "MultiHop runs last"):
            self._cell("multihop_rag", generator=PHI)
        self.assertEqual(len(self.commands), 2)

    def test_a_finished_cell_is_not_run_again(self):
        self._cell()
        self._cell()
        self.assertEqual(len(self.commands), 1)

    def test_only_the_sweeps_benchmarks_and_generators(self):
        with self.assertRaisesRegex(SystemExit, "the sweep runs narrativeqa and multihop_rag"):
            self._cell("hotpotqa")
        for generator in (settings.READER, "Qwen/Qwen2.5-0.5B-Instruct"):
            with self.assertRaisesRegex(SystemExit, "not one of the sweep's generators"):
                self._cell(generator=generator)
        self.assertEqual(self.commands, [])

    def test_only_a_registered_sweep_and_v20s_model(self):
        with mock.patch.dict("scripts.report_generators.SWEEP_CODE", {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "not a registered sweep"):
                self._cell()
        with mock.patch.dict("scripts.report_ours.SAME_OUTPUTS", {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "Check out its commit"):
                self._cell()
        self.assertEqual(self.commands, [])

    def test_a_rounds_bank_takes_no_generator(self):
        with mock.patch.dict("scripts.report_ours.FROZEN_CODE", {"ours_v99_generators": self.code}), \
                mock.patch.dict("scripts.report_generators.SWEEP_CODE", {}, clear=True), \
                self.assertRaisesRegex(SystemExit, "not a registered sweep"):
            self._cell()
        self.assertEqual(self.commands, [])


class TestSweepTrees(unittest.TestCase):
    """Ahead of a sweep cell, every tree it will read, built with the builder alone, read warm."""

    def _build(self, cache, generator, units_cls, out=None):
        def never(*args, **kwargs):
            raise AssertionError("bge-m3 or the reranker loaded while building trees")

        with ours_fakes(cache, FakeReader(DECOMPOSITIONS, ANSWERS)), \
                mock.patch.dict("src.eval.runner.BENCHMARK_REGISTRY", {"narrativeqa": units_cls}), \
                mock.patch("ours.model.add_second_vectors", never), \
                mock.patch("ours.model.cross_encoder_scores", never), \
                contextlib.redirect_stdout(out or io.StringIO()):
            return dev_ours.build_own_trees("narrativeqa", check_env=lambda lockfile: None,
                                            generator=generator)

    def test_each_built_tree_shows_one_of_its_summaries(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as td:
            self._build(Path(td), QWEN3B, _MixedUnits, out=out)
        shown = [line for line in out.getvalue().splitlines() if "a summary: " in line]
        self.assertEqual(len(shown), 2)
        # The fake summariser's text: a cluster's first twenty words.
        self.assertTrue(all(len(line.split("a summary: ", 1)[1]) > 10 for line in shown))

    def test_every_unit_is_built_once_and_the_cell_reads_them_warm(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            first = self._build(cache, QWEN3B, _MixedUnits)
            again = self._build(cache, QWEN3B, _MixedUnits)
            self.assertEqual(first, {"units": 2, "warm": 0, "built": 2})
            self.assertEqual(again, {"units": 2, "warm": 2, "built": 0})
            self.assertEqual(len(tree_dirs(cache, settings.SWEEP_TREE_NAMESPACE)), 1)
            self.assertEqual(len(tree_dirs(cache, settings.TREE_NAMESPACE)), 1)
            self.assertFalse((cache / M4_SUBSTRATE_NAMESPACE).exists())
            with ours_fakes(cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
                system = om.OursSystem(config=dev_ours.generator_config(QWEN3B))
                for unit in _MixedUnits().units:
                    system.index_items(unit.corpus)
                    self.assertTrue(system.tree_cache_hit)

    def test_the_revision_checked_is_the_generators(self):
        seen = []
        with tempfile.TemporaryDirectory() as td, ours_fakes(td, FakeReader(DECOMPOSITIONS, ANSWERS)), \
                mock.patch("ours.model.check_model_revisions", lambda *a: seen.append(a)), \
                mock.patch.dict("src.eval.runner.BENCHMARK_REGISTRY", {"narrativeqa": _ShortUnits}), \
                contextlib.redirect_stdout(io.StringIO()):
            dev_ours.build_own_trees("narrativeqa", check_env=lambda lockfile: None, generator=PHI)
        self.assertEqual(seen, [(PHI,)])

    def test_the_readers_trees_are_still_ours_own_only(self):
        with tempfile.TemporaryDirectory() as td:
            counts = self._build(Path(td), None, _MixedUnits)
            self.assertEqual(counts, {"units": 1, "warm": 0, "built": 1})
            self.assertFalse((Path(td) / M4_SUBSTRATE_NAMESPACE).exists())

    def test_a_model_outside_the_sweep_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "not one of the sweep's generators"):
            dev_ours.build_own_trees("narrativeqa", check_env=lambda lockfile: None,
                                     generator="Qwen/Qwen2.5-0.5B-Instruct")


if __name__ == "__main__":
    unittest.main()
