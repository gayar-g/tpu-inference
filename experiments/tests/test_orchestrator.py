import unittest
from unittest.mock import patch
import sys
import os
import yaml

# Add the directory containing run_sweep_orchestrator to path so we can import it
# Since this script will be in `experiments/tests/`, we need to go up one level
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_sweep_orchestrator

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_config.yaml")
        
    def _write_config(self, config_dict):
        with open(self.config_path, "w") as f:
            yaml.dump(config_dict, f)
            
    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    @patch('run_sweep_orchestrator.subprocess.run')
    @patch('run_sweep_orchestrator.argparse.ArgumentParser.parse_args')
    @patch('run_sweep_orchestrator.os.path.exists') 
    def test_single_targeted_trace(self, mock_exists, mock_parse_args, mock_run):
        config = {
            "model": "google/gemma-4-12b-it",
            "sweep_matrix": {
                "batch_size": [1, 4],
                "input_len": [512, 1024],
                "output_len": [64],
            },
            "trace_configs": [
                {"batch_size": 4, "input_len": 1024, "output_len": 64}
            ]
        }
        self._write_config(config)
        
        class DummyArgs:
            config = self.config_path
            result_dir = "/tmp/mock_results"
        mock_parse_args.return_value = DummyArgs()
        mock_exists.return_value = False 
        
        run_sweep_orchestrator.main()
        
        # 2 batches x 2 input_lens * 1 output_len = 4 runs
        self.assertEqual(mock_run.call_count, 4)
        
        traces_found = 0
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            is_targeted = ("--batch-size" in cmd and cmd[cmd.index("--batch-size") + 1] == "4" and 
                          "--input-len" in cmd and cmd[cmd.index("--input-len") + 1] == "1024")
                          
            if "--trace" in cmd:
                traces_found += 1
                self.assertTrue(is_targeted, "Trace flag was applied to an untargeted coordinate!")
                
        self.assertEqual(traces_found, 1, "Exactly one run should have been traced.")

    @patch('run_sweep_orchestrator.subprocess.run')
    @patch('run_sweep_orchestrator.argparse.ArgumentParser.parse_args')
    @patch('run_sweep_orchestrator.os.path.exists') 
    def test_multiple_targeted_traces(self, mock_exists, mock_parse_args, mock_run):
        # We want to trace BOTH Batch 4/Input 1024 AND Batch 1/Input 512
        config = {
            "model": "google/gemma-4-12b-it",
            "sweep_matrix": {
                "batch_size": [1, 4],
                "input_len": [512, 1024],
                "output_len": [64],
            },
            "trace_configs": [
                {
                    "batch_size": 4, 
                    "input_len": 1024, 
                    "output_len": 64,
                    "jax_advanced_configuration": {"tpu_enable_periodic_counter_sampling": True}
                },
                {"batch_size": 1, "input_len": 512, "output_len": 64}
            ]
        }
        self._write_config(config)
        
        class DummyArgs:
            config = self.config_path
            result_dir = "/tmp/mock_results"
        mock_parse_args.return_value = DummyArgs()
        mock_exists.return_value = False 
        
        run_sweep_orchestrator.main()
        
        self.assertEqual(mock_run.call_count, 4)
        
        traces_found = 0
        advanced_config_found = False
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            is_target_1 = ("--batch-size" in cmd and cmd[cmd.index("--batch-size") + 1] == "4" and 
                           "--input-len" in cmd and cmd[cmd.index("--input-len") + 1] == "1024")
            
            is_target_2 = ("--batch-size" in cmd and cmd[cmd.index("--batch-size") + 1] == "1" and 
                           "--input-len" in cmd and cmd[cmd.index("--input-len") + 1] == "512")
                           
            if "--trace" in cmd:
                traces_found += 1
                self.assertTrue(is_target_1 or is_target_2, "Trace flag applied to an untargeted coordinate!")
                
                if "--jax-advanced-configuration" in cmd:
                    advanced_config_found = True
                
        self.assertEqual(traces_found, 2, "Exactly TWO runs should have been traced.")
        self.assertTrue(advanced_config_found, "The Advanced configuration JSON flag was missing!")

if __name__ == '__main__':
    unittest.main(verbosity=2)
