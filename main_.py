from pymoo.util.display import disp_multi_objective


class NET:
    def __init__(self):
        self.func_display_attrs = disp_multi_objective
        self.pop_size = None

        self.elitist_archive_X = []
        self.elitist_archive_hashX = []
        self.elitist_archive_F = []

        self.dpfs = []
        self.no_eval = []

        self.no_evaluations = 0

    def _solve