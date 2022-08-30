import sys
sys.path.insert(0, ".")

import numpy as np
from utils.dataset import Dataset
from utils.styled_plot import plt


def calculate_ice(model, X, s):
    """
    Takes the input data and expands the dimensions from (num_instances, num_features) to (num_instances,
    num_instances, num_features). For the current instance i and the selected feature index s, the
    following equation is ensured: X_ice[i, :, s] == X[i, s].
    
    Parameters:
        model: Classifier which can call a predict method.
        X (np.array with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        
    Returns:
        X_ice (np.array with shape (num_instances, num_instances, num_features)): Changed input data w.r.t. x_s.
        y_ice (np.array with shape (num_instances, num_instances)): Predicted data.
    """

    shape = X.shape
    X_ice = np.zeros((shape[0], *shape))
    y_ice = np.zeros((shape[0], shape[0]))

    # Iterate over data points
    for i, _ in enumerate(X):
        X_copy = X.copy()
        
        # Intervention
        # Take the value of i-th data point and set it to all others
        # We basically fix the value
        X_copy[:, s] = X_copy[i, s]
        X_ice[i] = X_copy
        
        # Then we do a prediction with the new data
        y_ice[i] = model.predict(X_copy)
        
    return X_ice, y_ice


def prepare_ice(model, X, s, centered=False):
    """
    Uses `calculate_ice` to retrieve plot data.
    
    Parameters:
        model: Classifier which can call a predict method.
        X (np.array with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used or not.
        
    Returns:
        all_x (list or 1D np.ndarray): List of lists of the x values.
        all_y (list or 1D np.ndarray): List of lists of the y values.
            Each entry in `all_x` and `all_y` represents one line in the plot.
    """
    
    X_ice, y_ice = calculate_ice(model, X, s)
    
    all_x = []
    all_y = []

    for i in range(X.shape[0]):
        x = X_ice[:, i, s]
        y = y_ice[:, i]
            
        # We have to sort x because they might be not
        # in the right order
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
            
        # Or all zero centered (c-ICE)
        if centered:
            y = y - y[0]
        
        all_x.append(x)
        all_y.append(y)
            
    return all_x, all_y


def plot_ice(model, dataset, X, s, centered=False):
    """
    Creates a plot object and fills it with the content of `prepare_ice`.
    Note: `show` method is not called.
    
    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Used dataset to train the model. Used to receive the labels.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used or not.
        
    Returns: 
        plt (matplotlib.pyplot or utils.styled_plot.plt)
    """
    
    all_x, all_y = prepare_ice(model, X, s, centered)
    
    plt.figure()
    plt.xlabel(dataset.get_input_labels(s))
    plt.ylabel(dataset.get_output_label())

    # Now do the plotting
    for x, y in zip(all_x, all_y):
        plt.plot(x, y, linewidth=0.25, alpha=0.2, color="black")
        
    return plt
        

def prepare_pdp(model, X, s):
    """
    Uses `calculate_ice` to retrieve plot data for PDP.
    
    Parameters:
        model: Classifier which can call a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        
    Returns:
        x (list or 1D np.ndarray): x values of the PDP line.
        y (list or 1D np.ndarray): y values of the PDP line.
    """
    
    _, y_ice = calculate_ice(model, X, s)
        
    # Take all x_s instance value
    x = X[:, s]
    # Alternatively use X_ice directly
    # x = X_ice[:, 0, s]
    
    y = []
    # Simply take all values and mean them
    for i in range(y_ice.shape[0]):
        y.append(np.mean(y_ice[i, :]))
            
    y = np.array(y)
        
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    return x, y


def plot_pdp(model, dataset, X, s):
    """
    Creates a plot object and fills it with the content of `prepare_pdp`.
    Note: `show` method is not called.
    
    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Used dataset to train the model. Used to receive the labels.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used or not.
        
    Returns: 
        plt (matplotlib.pyplot or utils.styled_plot.plt)
    """
    
    x, y = prepare_pdp(model, X, s)
    
    plt.figure()
    plt.xlabel(dataset.get_input_labels(s))
    plt.ylabel(dataset.get_output_label())

    # Now do the plotting
    plt.plot(x, y, linewidth=0.25, alpha=1, color="black")
        
    return plt


if __name__ == "__main__":
    dataset = Dataset("wheat_seeds", [5,6,7], [2], normalize=True, categorical=False)
    (X_train, y_train), (X_test, y_test) = dataset.get_data()
    
    from sklearn import ensemble
    model = ensemble.RandomForestRegressor()
    model.fit(X_train, y_train)
    X = dataset.X
    s = 1

    print("Run `calculate_ice` ...")
    calculate_ice(model, X, s)
    
    print("Run `prepare_ice` ...")
    prepare_ice(model, X, s, centered=False)
    
    print("Run `plot_ice` ...")
    plt = plot_ice(model, dataset, X, s, centered=False)
    plt.show()
    
    print("Run `plot_ice` with centered=True ...")
    plt = plot_ice(model, dataset, X, s, centered=True)
    plt.show()
    
    print("Run `prepare_pdp` ...")
    prepare_pdp(model, X, s)
    
    print("Run `plot_pdp` ...")
    plt = plot_pdp(model, dataset, X, s)
    plt.show()
