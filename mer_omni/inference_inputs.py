"""Explicit inference-only overflow handling with native feature-length receipts."""


class InferenceInputAudit:
    def __init__(self, frontend, overflow=None):
        if overflow not in (None, 'error', 'truncate_right'):
            raise ValueError('Unknown inference sequence overflow policy')
        if overflow is not None and frontend.input_mode != 'zero_pad':
            raise ValueError('Inference overflow overrides require zero_pad inputs')
        self.frontend = frontend
        self.original_overflow = frontend.sequence_overflow
        self.policy = dict(input_mode=frontend.input_mode,
                           sequence_lengths=list(frontend.sequence_lengths),
                           sequence_overflow=overflow or self.original_overflow)
        self.processed_samples = 0
        self.affected = []
        self.active_id = None
        self.observed = False

    def __enter__(self):
        # Configuration and calibrated tensor shapes remain unchanged; only this
        # live inference adapter receives the explicitly selected overflow rule.
        self.frontend.sequence_overflow = self.policy['sequence_overflow']
        self.handle = self.frontend.register_forward_pre_hook(self._capture)
        return self

    def begin_sample(self, sample_id):
        if self.active_id is not None and not self.observed:
            raise RuntimeError('Previous sample did not pass through the spectral frontend')
        self.active_id, self.observed = sample_id, False

    def _capture(self, frontend, arguments):
        if self.active_id is None or self.observed:
            raise RuntimeError('Expected one spectral frontend call per inference sample')
        sequences = arguments[0]
        if len(sequences) != 3 or any(len(values) != 1 for values in sequences):
            raise ValueError('Inference length audit requires one sample and three modalities')
        native = [int(values[0].shape[0]) for values in sequences]
        if frontend.input_mode == 'zero_pad':
            discarded = [max(0, n-limit) for n, limit in zip(native, frontend.sequence_lengths)]
            if any(discarded):
                self.affected.append(dict(id=self.active_id, native_lengths=native,
                    retained_lengths=[n-d for n, d in zip(native, discarded)],
                    discarded_positions=discarded))
        self.processed_samples += 1
        self.observed = True

    def report(self):
        if self.active_id is not None and not self.observed:
            raise RuntimeError('Last sample did not pass through the spectral frontend')
        return dict(policy=self.policy, modality_order=['text', 'video', 'audio'],
                    processed_samples=self.processed_samples, affected_samples=len(self.affected),
                    affected=self.affected)

    def __exit__(self, *unused):
        self.handle.remove()
        self.frontend.sequence_overflow = self.original_overflow
