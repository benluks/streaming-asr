import torch
from speechbrain.inference.ASR import StreamingASR, ASRStreamingContext
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.decoders.transducer import TransducerGreedySearcherStreamingContext

CHUNK_SIZE = 8  # Adjust for different chunk durations
CHUNK_LEFT_CONTEXT = 2  # Number of previous chunks used as context
CHUNK_FRAMES = 639

chunk_len = CHUNK_SIZE * CHUNK_FRAMES


def load_asr_model(chunk_size, chunk_left_context) -> StreamingASR:
    """
    Loads the SpeechBrain ASR streaming model.

    Returns:
        StreamingASR: Initialized ASR model.
    """
    asr_model = StreamingASR.from_hparams(
        "speechbrain/asr-streaming-conformer-librispeech"
    )
    context = asr_model.make_streaming_context(
        DynChunkTrainConfig(chunk_size, chunk_left_context)
    )

    return asr_model, context


def encode_chunk(
    asr_model: StreamingASR,
    context: ASRStreamingContext,
    chunk: torch.Tensor,
    chunk_len: torch.Tensor,
) -> torch.Tensor:
    """
    Encodes a chunk of audio.

    Args:
        asr_model (StreamingASR): The ASR model instance.
        context (ASRStreamingContext): The streaming context.
        chunk (torch.Tensor): The audio chunk to encode.

    Returns:
        torch.Tensor: Encoded representation of the audio chunk.
    """

    return asr_model.encode_chunk(context, chunk, chunk_len)


def transcribe_chunk(
    asr_model: StreamingASR, context: ASRStreamingContext, chunk: torch.Tensor
) -> str:
    """
    Transcribes an audio chunk using ASR.

    Args:
        asr_model (StreamingASR): The ASR model instance.
        context (ASRStreamingContext): The streaming context.
        chunk (torch.Tensor): The audio chunk.

    Returns:
        str: Transcribed text.
    """
    words = asr_model.transcribe_chunk(context, chunk)
    return words


def decode_encoding(asr_model, context, encoding):

    # for testing. to be removed shortly
    # tokens = asr_model.hparams.decoding_function(encoding, context.decoder_context)
    # should be same as following:
    tokens, *_, hidden = asr_model.hparams.Greedysearcher.transducer_greedy_decode(
        encoding, context.decoder_context.hidden, True
    )
    context.decoder_context.hidden = hidden

    if context.tokenizer_context is None:
        context.tokenizer_context = [
            asr_model.hparams.make_tokenizer_streaming_context()
            for _ in range(len(tokens))
        ]

    words = [
        asr_model.hparams.tokenizer_decode_streaming(
            asr_model.hparams.tokenizer, cur_tokens, context.tokenizer_context[i]
        )
        for i, cur_tokens in enumerate(tokens)
    ]
    return words

    # return tokens


def get_logits_from_encoding(
    asr_model: StreamingASR,
    encoding: torch.Tensor,
    hidden_state: tuple[torch.Tensor, torch.Tensor],
):
    """
    Decodes the encoding to obtain the logits.
    Args:
        asr_model (StreamingASR): The ASR model instance.
        context (ASRStreamingContext): The streaming context.
        encoding (torch.Tensor): The encoded representation of the audio chunk.
        hidden_state tuple(torch.Tensor, torch.Tensor): The hidden state of the decoder.

    Implementation from: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.decoders.transducer.html#speechbrain.decoders.transducer.TransducerBeamSearcher.transducer_greedy_decode
    """
    decoder = asr_model.hparams.Greedysearcher

    logits = torch.tensor([])
    prediction = []

    # prepare BOS = Blank for the Prediction Network (PN)
    input_PN = (
        torch.ones(
            (encoding.size(0), 1),
            device=encoding.device,
            dtype=torch.int32,
        )
        * decoder.blank_id
    )

    if hidden_state is None:
        # First forward-pass on PN
        out_PN, hidden = decoder._forward_PN(input_PN, decoder.decode_network_lst)
    else:
        out_PN, hidden = hidden_state

    # For each time step
    for t_step in range(encoding.size(1)):
        # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
        log_probs = decoder._joint_forward_step(
            encoding[:, t_step, :].unsqueeze(1).unsqueeze(1),
            out_PN.unsqueeze(1),
        )

        predicted_log_prob, predicted_idx = torch.max(
            log_probs.squeeze(1).squeeze(1), dim=1
        )

        # run forward on PN if prediction is not blank, else don't update hidden
        try:
            is_hyp_updated = predicted_idx.item() != decoder.blank_id
        except RuntimeError:
            print(f"`get_logits_from_encoding` not yet implemented for batch decoding")
            break

        if is_hyp_updated:
            prediction.append(predicted_idx.item())
            input_PN[0] = predicted_idx
            out_PN, hidden = decoder._forward_PN(
                input_PN, decoder.decode_network_lst, hidden
            )

        logits = torch.cat((logits, log_probs.squeeze(1)), dim=1)

    return logits, (out_PN, hidden), [prediction]


def streaming_decode_step(
    asr_model, encoding, context: TransducerGreedySearcherStreamingContext
):
    logits, hidden_state, prediction = get_logits_from_encoding(
        asr_model, encoding, context.hidden
    )
    context.hidden = hidden_state
    return logits, prediction


def decode_chunk(asr_model, context, encoding):
    logits, tokens = streaming_decode_step(asr_model, encoding, context.decoder_context)

    # initialize token context for real now that we know the batch size
    if context.tokenizer_context is None:
        context.tokenizer_context = [
            asr_model.hparams.make_tokenizer_streaming_context()
            for _ in range(len(tokens))
        ]

    words = [
        asr_model.hparams.tokenizer_decode_streaming(
            asr_model.hparams.tokenizer, cur_tokens, context.tokenizer_context[i]
        )
        for i, cur_tokens in enumerate(tokens)
    ]

    return logits, tokens, words


def transcribe(asr_model, context, chunk, chunk_len=None):

    if chunk_len is None:
        chunk_len = torch.ones((chunk.size(0),))

    chunk = chunk.float()
    chunk, chunk_len = chunk.to(asr_model.device), chunk_len.to(asr_model.device)

    x = asr_model.encode_chunk(context, chunk, chunk_len)
    *_, words = decode_chunk(asr_model, context, x)

    return words
