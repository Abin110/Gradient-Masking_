import random
from reedsolo import RSCodec, ReedSolomonError
import numpy as np
from torchvision import transforms
from PIL import Image
import torch


class MessageEcodeDecode:
    def __init__(self, config):
        self.ecc_size = config['ecc_size']
        self.host_image_size = config['host_image_size']
        self.message_type = config.get('message_type', 'image')  # 默认为图像处理

    def seq_to_array(self, binary_sequence, width, length):

        array_size = width * length
        truncated_sequence = binary_sequence[:array_size]
        filled_sequence = truncated_sequence.ljust(array_size, '0')
        binary_array = np.array(list(map(int, filled_sequence))).reshape(width, length)
        return binary_array

    def array_to_seq(self, binary_array):
        flattened_array = binary_array.flatten()
        binary_sequence = ''.join(map(str, flattened_array))
        return binary_sequence

    def encode(self, message_path):
        if self.message_type == 'image':
            return self.encode_image(message_path)
        elif self.message_type == 'text':
            return self.encode_text(message_path)
        else:
            raise ValueError("Unsupported message type. Use 'image' or 'text'.")

    def decode(self, encoded_data, output_path):
        if self.message_type == 'image':
            return self.decode_image(encoded_data, output_path)
        elif self.message_type == 'text':
            return self.decode_text(encoded_data, output_path)
        else:
            raise ValueError("Unsupported message type. Use 'image' or 'text'.")

    def encode_image(self, image_path):
        with open(image_path, 'rb') as f:
            image_bytes = bytearray(f.read())

        rs = RSCodec(self.ecc_size)
        encoded_bytes = rs.encode(image_bytes)

        binary_data = ''.join(format(byte, '08b') for byte in encoded_bytes)
        return binary_data

    def encode_text(self, text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text_bytes = text.encode('utf-8')
        rs = RSCodec(self.ecc_size)
        encoded_bytes = rs.encode(text_bytes)

        binary_data = ''.join(format(byte, '08b') for byte in encoded_bytes)
        return binary_data

    def enmask(self, message_path):
        binary_flow = self.encode(message_path)
        width, length = self.host_image_size
        mask_ary = self.seq_to_array(binary_flow, width=width, length=length)
        mask_tensor = transforms.ToTensor()(mask_ary).to('cuda')
        return mask_tensor

    def demask(self, perturbation, output_path):
        width, length = self.host_image_size
        perturbation[perturbation != 0] = 1
        perturbation = perturbation.int()
        mask_arr = np.array(perturbation[0].cpu().detach())
        binary_flow = self.array_to_seq(mask_arr)

        byte_sequence = [int(binary_flow[i:i + 8], 2) for i in range(0, len(binary_flow), 8)]
        while byte_sequence[-1] == 0:
            byte_sequence.pop()

        encoded_bytes = bytearray(byte_sequence)
        rs = RSCodec(self.ecc_size)
        decoded_bytes = rs.decode(encoded_bytes)[0]

        if self.message_type == 'image':
            with open(output_path, 'wb') as f:
                f.write(decoded_bytes)
        elif self.message_type == 'text':
            decoded_text = decoded_bytes.decode('utf-8', errors='ignore')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(decoded_text)

        return decoded_bytes



