import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype


class TritonClientWrapper:
    def __init__(self, url, verbose=False):
        self._url = url
        self._client = httpclient.InferenceServerClient(
            url=url, verbose=verbose)

    def _predict(self, model, _inputs):
        _model_metadata = self._client.get_model_metadata(model)
        #_model_config = self._client.get_model_config(model)

        _input_signature = {
            x['name']: {'shape': x['shape'],
                        'datatype': x['datatype']}
            for x in _model_metadata['inputs']
        }
        _output_signature = {
            x['name']: {'shape': x['shape'],
                        'datatype': x['datatype']}
            for x in _model_metadata['outputs']
        }

        missing_inputs = [k for k in _input_signature if k not in _inputs]
        if len(missing_inputs):
            raise ValueError(
                f'Missing inputs for prediction - {missing_inputs}')

        # TODO Validate shape

        model_inputs = [
            httpclient.InferInput(
                x, _inputs[x].shape, _input_signature[x]['datatype'])
            for x in _input_signature
        ]
        for i, x in enumerate(_input_signature):
            # Get np dtype
            _dtype = triton_to_np_dtype(_input_signature[x]['datatype'])
            # Set np array as input
            model_inputs[i].set_data_from_numpy(_inputs[x].astype(_dtype))

        model_outputs = [
            httpclient.InferRequestedOutput(x) for x in _output_signature
        ]

        _outputs = self._client.infer(
            model,
            model_inputs,
            outputs=model_outputs)

        return {
            k: _outputs.as_numpy(k)
            for k in _output_signature
        }

    def predict(self, model, _inputs):
        outputs = self._predict(model, _inputs)
        return outputs


if __name__ == '__main__':
    # https://github.com/triton-inference-server/client/blob/main/src/python/examples/image_client.py
    import argparse
    import sys
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--model', required=True)
    parser.add_argument('--url', default="localhost:8000")
    args = parser.parse_args()

    try:
        client = TritonClientWrapper(args.url)
    except Exception as e:
        print('Client creation failed - ', str(e))
        sys.exit(1)

    image_data = []
    with Image.open(args.image) as im:
        img = np.array(im)
        img = img[..., ::-1]
        #img = img.transpose(2, 0, 1)
        image_data.append(img)

    outputs = client.predict(args.model, {
        'image': image_data[0]
    })
    print(outputs)
