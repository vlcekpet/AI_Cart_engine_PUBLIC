import numpy as np


class ShallowNNRandom:
    """
    Jednoduchá 'shallow' neuronová síť s náhodnými parametry.
    - 4 vstupy
    - 1 skrytá vrstva
    - 3 výstupy
    - bez učení (parametry se jen náhodně inicializují)
    """

    def __init__(self, n_inputs=4, n_hidden=8, n_outputs=3,
                 rng=None, weight_scale=0.1):
        """
        n_inputs   : počet vstupů (feature)
        n_hidden   : počet neuronů ve skryté vrstvě
        n_outputs  : počet výstupů (např. 3 třídy)
        rng        : np.random.Generator (volitelné)
        weight_scale : směrodatná odchylka pro náhodné váhy
        """
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        # Parametry 1. vrstvy: (n_hidden, n_inputs) a (n_hidden,)
        self.W1 = self.rng.normal(loc=0.0, scale=weight_scale,
                                  size=(n_hidden, n_inputs))
        self.b1 = self.rng.normal(loc=0.0, scale=weight_scale,
                                  size=(n_hidden,))

        # Parametry 2. vrstvy: (n_outputs, n_hidden) a (n_outputs,)
        self.W2 = self.rng.normal(loc=0.0, scale=weight_scale,
                                  size=(n_outputs, n_hidden))
        self.b2 = self.rng.normal(loc=0.0, scale=weight_scale,
                                  size=(n_outputs,))

    @staticmethod
    def _relu(z):
        return np.maximum(z, 0.0)

    @staticmethod
    def _softmax(z):
        """
        Softmax po řádcích.
        z: shape (n_samples, n_outputs)
        """
        z = z - np.max(z, keepdims=True)  # numerická stabilita
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, keepdims=True)

    def forward(self, X):
        """
        X : array shape (n_samples, 4)
        Návrat: raw výstupy shape (n_samples, 3)
        """
        X = np.asarray(X, dtype=float)

        # 1. vrstva: X -> hidden
        z1 = X @ self.W1.T + self.b1  # (n_samples, n_hidden)
        h1 = self._relu(z1)

        # 2. vrstva: hidden -> output scores
        z2 = h1 @ self.W2.T + self.b2  # (n_samples, n_outputs)
        return z2

    def predict_proba(self, X):
        """
        Vrací softmax pravděpodobnosti (shape (n_samples, 3)).
        """
        scores = self.forward(X)
        return self._softmax(scores)

    def predict_class(self, X):
        """
        Vrací index nejpravděpodobnější třídy (0, 1 nebo 2).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    # Příklad použití:

    # Vytvoříme síť
    model = ShallowNNRandom(n_inputs=4, n_hidden=8, n_outputs=3)

    # Náhodná data: 5 vzorků, každý má 4 vstupy
    X = np.random.randn(5, 4)

    # Výstupní "scores" a pravděpodobnosti
    scores = model.forward(X)
    probs = model.predict_proba(X)
    y_pred = model.predict_class(X)

    print("Vstupy X:\n", X)
    print("Scores (raw výstupy):\n", scores)
    print("Softmax pravděpodobnosti:\n", probs)
    print("Predikované třídy:", y_pred)