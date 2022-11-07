import logging
import multiprocessing as mp

import torch
from torchaudio.io import StreamReader

logger = logging.getLogger(__file__)


class Streaming(nn.Module):
    def __init__(self, model, melspec, window_size=7500, step_size=3750, is_half=True):
        super().__init__()


        self.window_size = window_size
        self.step_size = step_size

        self.is_half = is_half
        self.model = model if not is_half else model.half()
        config = self.model.config
        self.device = config.device

        self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=config.n_mels
        ).to(config.device)


        self.buffer = torch.zeros([1, 0])
        self.gru_outputs = torch.zeros([1, 0, 1]).to(self.device)
        self.hidden = torch.zeros([1, 0]).to(self.device)

        self.prev_proba = 0.

    def forward(self, chunk):
        assert len(chunk.size()) == 2
        max_proba = 0.

        if self.buffer.size()[-1] < self.window_size:
            need_len = self.window_size - self.buffer.size()[-1]

            self.buffer = torch.cat([self.buffer, chunk[:, :need_len]], dim=1)
            chunk = chunk[:, need_len:]

            if self.buffer.size()[-1] == self.window_size:
                max_proba = self.model_forward(self.buffer)
            else:
                return self.prev_proba

        chunks = torch.split(chunk, self.step_size, dim=-1)
        for chunk in chunks:

            self.buffer = torch.cat([self.buffer[:, self.step_size:], chunk], dim=1)

            if self.buffer.size()[-1] == self.window_size:
                cur_proba = self.model_forward(self.buffer)
                max_proba = max(max_proba, cur_proba)

        self.prev_proba = max_proba
        return max_proba


    def model_forward(self, x):

        if self.gru_outputs is not None:
            x = x[:, -self.step_size:]

        x = x.to(self.device)
        x = torch.log(self.melspec(x).clamp_(min=1e-9, max=1e9))
        x = x if not self.is_half else x.half()
        input = x.unsqueeze(dim=1)
        conv_output = self.model.conv(input).transpose(-1, -2)

        hidden = None if self.hidden.size()[-1] == 0 else self.hidden
        gru_outputs, hidden = self.model.gru(conv_output, hidden)
        self.hidden = hidden

        if self.gru_outputs.size()[1] != 0:
            gru_outputs = torch.cat([self.gru_outputs[:, self.step_size:, :], gru_outputs], dim=1)

        self.gru_outputs = gru_outputs

        contex_vector = self.model.attention(gru_outputs)
        output = self.model.classifier(contex_vector)
        return F.softmax(output, dim=-1)[0][1].item()


def audio_stream(queue: mp.Queue):
    """
    Learn more about how to install and use streaming audio here
    https://pytorch.org/audio/stable/tutorials/streaming_api2_tutorial.html
    """

    streamer = StreamReader(src=":0", format="avfoundation")
    streamer.add_basic_audio_stream(frames_per_chunk=7500, sample_rate=16000)
    stream_iterator = streamer.stream(-1, 1)

    logger.info("Start audio streaming")
    while True:
        (chunk_,) = next(stream_iterator)
        logger.info("Put chunk to queue")
        queue.put(chunk_)


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Streaming")
    args.add_argument(
            "-c",
            "--ckpt",
            default='stream_jit.pt',
            type=str,
            help="model file path (default: kws.pth)",
        )

    args = args.parse_args()
    model = torch.jit.load(args.ckpt).eval()

    ctx = mp.get_context("spawn")
    chunk_queue = ctx.Queue()
    streaming_process = ctx.Process(target=audio_stream, args=(chunk_queue,))

    streaming_process.start()
    while True:
        try:
            chunk = chunk_queue.get()
            chunk = chunk.view(1, -1)
            print(f"{chunk.shape=}")

            with torch.inference_mode():
                result = model(chunk)

            if result > 0.7:
                print("DETECTED KEY WORD")

        except KeyboardInterrupt:
            break
        except Exception as exc:
            raise exc

    streaming_process.join()
