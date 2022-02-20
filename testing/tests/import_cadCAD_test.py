import unittest, os, subprocess, json

class JupyterServerTest(unittest.TestCase):
    def test_row_count(self):
        command = f'jupyter nbconvert --to=notebook --ExecutePreprocessor.enabled=True {os.getcwd()}/testing/tests/import_prima.ipynb'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
        json_path = f'{os.getcwd()}/testing/tests/prima_memory_address.json'
        memory_address = json.load(open(json_path))['memory_address']
        self.assertEqual(type(memory_address) == str, True, "prima is not importable by jupyter server")

if __name__ == '__main__':
    unittest.main()