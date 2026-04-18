from __future__ import annotations

from pathlib import Path
import unittest


class TrainingRequirementsDocsTests(unittest.TestCase):
    def test_root_requirements_and_readme_document_training_setup(self):
        repo_root = Path(__file__).resolve().parents[1]
        requirements_path = repo_root / 'requirements.txt'
        readme_path = repo_root / 'README.md'
        training_flow_path = repo_root / 'OrchRL_Training_Flow.md'
        experiment_launcher_path = repo_root / 'experiments/search_mas/run_train_e2e.sh'
        experiment_config_path = repo_root / 'experiments/search_mas/train.yaml'
        experiment_readme_path = repo_root / 'experiments/search_mas/README.md'
        inference_config_path = repo_root / 'mas_apps/search/configs/inference.yaml'
        manual_clean_ray_path = repo_root / 'scripts/clean_ray.py'

        self.assertTrue(requirements_path.is_file(), 'missing repository-root requirements.txt')
        requirements_text = requirements_path.read_text(encoding='utf-8')
        readme_text = readme_path.read_text(encoding='utf-8')
        training_flow_text = training_flow_path.read_text(encoding='utf-8')
        experiment_readme_text = experiment_readme_path.read_text(encoding='utf-8')
        experiment_launcher_text = experiment_launcher_path.read_text(encoding='utf-8')

        self.assertIn('-r mas_apps/search/requirements.txt', requirements_text)
        self.assertIn('codetiming', requirements_text)
        self.assertIn('python -m pip install -r requirements.txt', readme_text)
        self.assertIn('bash experiments/search_mas/run_train_e2e.sh', readme_text)
        self.assertIn('mas_apps/search/', readme_text)
        self.assertIn('retriever', readme_text.lower())
        self.assertIn('separate conda environment', readme_text.lower())
        self.assertTrue(experiment_launcher_path.is_file(), 'missing experiment train launcher')
        self.assertTrue(experiment_config_path.is_file(), 'missing experiment train config')
        self.assertTrue(experiment_readme_path.is_file(), 'missing experiment README')
        self.assertTrue(inference_config_path.is_file(), 'missing inference config')
        self.assertFalse(manual_clean_ray_path.exists(), 'unused manual Ray cleanup script should be removed')
        self.assertIn('mas_apps/search', training_flow_text)
        self.assertIn('experiments/search_mas/train.yaml', training_flow_text)
        self.assertIn('mas_apps/search/configs/inference.yaml', training_flow_text)
        self.assertIn('outputs/training_runs/<experiment_name>/<run_id>/checkpoints', readme_text)
        self.assertIn('outputs/training_runs/<experiment_name>/<run_id>/trajectories', readme_text)
        self.assertIn('outputs/logs/', readme_text)
        self.assertIn('training.train_data_path', readme_text)
        self.assertIn('training.val_data_path', readme_text)
        self.assertIn('${training.data_root_dir}/train.parquet', readme_text)
        self.assertIn('${training.data_root_dir}/test.parquet', readme_text)
        self.assertIn('does not retain the full validation trajectory set in memory', readme_text)
        self.assertIn('created automatically', readme_text.lower())
        self.assertIn('outputs/training_runs/<experiment_name>/<run_id>/checkpoints', training_flow_text)
        self.assertIn('outputs/training_runs/<experiment_name>/<run_id>/trajectories', training_flow_text)
        self.assertIn('outputs/logs/', training_flow_text)
        self.assertIn('training.train_data_path', training_flow_text)
        self.assertIn('training.val_data_path', training_flow_text)
        self.assertIn('training.validate_batch_size', training_flow_text)
        self.assertIn('${training.data_root_dir}/train.parquet', training_flow_text)
        self.assertIn('${training.data_root_dir}/test.parquet', training_flow_text)
        self.assertIn('validation/sample_avg_reward', training_flow_text)
        self.assertIn('role_specific', training_flow_text)
        self.assertIn('role_sharing', training_flow_text)
        self.assertNotIn('specialization: full', training_flow_text)
        self.assertNotIn('specialization: prompt', training_flow_text)
        self.assertIn('outputs/training_runs/<experiment_name>/<run_id>/checkpoints', experiment_readme_text)
        self.assertIn('outputs/training_runs/<experiment_name>/<run_id>/trajectories', experiment_readme_text)
        self.assertIn('outputs/logs/', experiment_readme_text)
        self.assertIn('training.validate_batch_size', experiment_readme_text)
        self.assertIn('${training.data_root_dir}/train.parquet', experiment_readme_text)
        self.assertIn('${training.data_root_dir}/test.parquet', experiment_readme_text)
        self.assertIn('does not retain the full validation trajectory set in memory', experiment_readme_text)
        self.assertIn('role_specific', experiment_readme_text)
        self.assertIn('role_sharing', experiment_readme_text)
        self.assertNotIn('`full`', experiment_readme_text)
        self.assertNotIn('`prompt`', experiment_readme_text)
        self.assertIn('role_specific', readme_text)
        self.assertIn('role_sharing', readme_text)
        self.assertIn('cfg.training.val_data_path', experiment_launcher_text)
        self.assertNotIn('cfg.training.mate.val_prompt_loader.path', experiment_launcher_text)
        self.assertNotIn('training.mate.val_prompt_loader.path', training_flow_text)
        self.assertNotIn('training.mate.val_prompt_loader.path', experiment_readme_text)
        self.assertNotIn('validation/env_state_success_rate', training_flow_text)
        self.assertNotIn('validation/agent_<role>/avg_turns', training_flow_text)
        self.assertNotIn('examples/mas_app/search', training_flow_text)
        self.assertNotIn('orchrl/config/search', training_flow_text)
        self.assertNotIn('mas_search_app.yaml', training_flow_text)


if __name__ == '__main__':
    unittest.main()
