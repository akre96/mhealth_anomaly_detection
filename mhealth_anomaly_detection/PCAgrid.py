# PCAgrid function from pcaPP R package wrapper
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr, PackageNotInstalledError
from rpy2.robjects import r, rinterface, vectors


class PCAgrid:
    def __init__(self, lib_loc: str | None = None, n_components: int = 2):
        # Install pcaPP R package if not already installed
        utils = importr("utils")
        try:
            pcaPP = importr("pcaPP", lib_loc=lib_loc)
        except PackageNotInstalledError:
            utils.chooseCRANmirror(ind=1)
            utils.install_packages("pcaPP", lib_loc=lib_loc)
            pcaPP = importr("pcaPP", lib_loc=lib_loc)
        self.pcaPP = pcaPP
        self.n_components = n_components
        self.trained = False

    # Convert numpy array to R matrix
    @staticmethod
    def dataToRMatrix(X: np.ndarray) -> vectors.FloatMatrix:
        r_data = rinterface.FloatSexpVector(X.flatten())
        return r["matrix"](r_data, nrow=X.shape[0], ncol=X.shape[1])

    def fit(self, X: np.ndarray | pd.DataFrame, *args, **kwargs) -> None:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        r_matrix = self.dataToRMatrix(X)
        self.model = self.pcaPP.PCAgrid(r_matrix, k=self.n_components)
        self.loadings = np.array(self.model.rx2("loadings"))
        self.components_ = self.loadings.T
        self.means = np.array(self.model.rx2("center"))
        self.trained = True

        return

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise ValueError(
                "PCAgrid model not trained. Please run .fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.dot(X - self.means, self.loadings)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError(
                "PCAgrid model not trained. Please run .fit() first."
            )

        return np.dot(x, self.loadings.T) + self.means
