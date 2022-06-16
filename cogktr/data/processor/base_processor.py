from ..datable import DataTable
class BaseProcessor:
    def __init__(self,debug=False):
        self.debug = debug
        pass

    def _process(self, data):
        pass

    def process_train(self, data):
        pass

    def process_dev(self, data):
        pass

    def process_test(self, data):
        pass

    def debug_process(self, data):
        if self.debug:
            debug_data = DataTable()
            for header in data.headers:
                debug_data[header] = data[header][:100]
            return debug_data
        return data
