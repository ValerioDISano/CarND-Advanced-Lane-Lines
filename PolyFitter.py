import numpy as np

class PolyFit(object):
    """
    Polynomial data fitting
    """
    def __init__(self, order):
        self.order = order
        self.current_fitting = None
        self._valid = False

        self.poly_data = tuple()

    def getFittedData(self):

        return self.poly_data

    def fit(self, first_coord_data, second_coord_data):
        
        if first_coord_data.size > 0 and second_coord_data.size > 0:
            self.current_fitting = np.polyfit(first_coord_data, second_coord_data, self.order)
            self._valid = True
        else:
            self._valid = False
            self.current_fitting = np.zeros((1,self.order+1))
        return self.current_fitting

    def fillPolyData(self, data_range):
    # Compute the points belonging to the fitted curve
        if self._valid:
            self.poly_data = tuple(self.compute_fitted_fcn_data(data_range))
        else:
            return
    
    def compute_fitted_fcn_data(self, data_range_first_coord):
        
        d_range = data_range_first_coord
        first_ax = np.linspace(d_range[0], d_range[1], d_range[1]-d_range[0]-1)
        second_ax = np.polyval(self.current_fitting, first_ax)

        return second_ax, first_ax
    
    def eval(self, data):

        return np.polyval(self.current_fitting, data)

    def isValid(self):
        return self._valid

    def invalidate(self):
        self._valid = False


    def getCurrentFit(self):
        return self.current_fitting
