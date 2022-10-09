class Representation:
    def __call__(self, audio_path, offset=0.0, duration=None):
        # NOTE: Should return tuple containing (rate: float, features: np.ndarray)
        raise NotImplementedError()
