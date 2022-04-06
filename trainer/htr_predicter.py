import torch
import os
from utils.page_writer import PageWriter
import utils.datafuncs
import utils.util


class Predicter:
    def __init__(self, model, config, inf_data_loader, device='cpu'):
        self.model = model

        self.config = config

        self.dataloaders = {'inf': inf_data_loader}

        self.device = device

        self.save_dir = str(config.save_dir)
        self.batch_size = config['inf_dataloader']['batch_size']

        if config.resume:
            self._load_checkpoint()
        else:
            assert config.resume, "Model missing from -r"

    def predict(self):
        print("Starting inference...")
        self.model.eval()
        dataset = self.dataloaders['inf'].dataset
        tokenizer = dataset.tokenizer

        all_predictions = []
        all_indices = []
        all_coords = []
        for i, (images, indxs, scales, vis_pad_lens) in enumerate(self.dataloaders['inf']):
            print(f"{i+1}/{len(self.dataloaders['inf'])}")
            with torch.no_grad():

                images = images.to(self.device)
                # words are <S>the_word<E>

                # Send in <S>the_word
                output, _, cross_attn, _ = self.model(images, target=None)
                # output shape: time, batch, output-size,
                # cross_attn shape: (block, batch, txt_len, vision_feat_len)
                predictions = torch.argmax(output, dim=-1)  # time, batch
                cross_attn = cross_attn[-1].cpu()  # batch, txt_len, vision_feat_len
                predictions = predictions.cpu()

            space_mask = predictions == tokenizer.token_to_id(' ') # time, batch
            space_mask = torch.cat((torch.ones(1, space_mask.shape[1]), space_mask[:-1]), dim=0)

            skip_tokens = [tokenizer.token_to_id(t) for t in ['<S>', '<E>', '<P>']]
            predictions = [''.join([tokenizer.id_to_token(t) for t in token if t not in skip_tokens]) for token in predictions.T]

            all_predictions.extend(predictions)
            all_indices.extend([int(indx) for indx in indxs])

            model_div = self.config['model']['feature_extractor']['width_div']
            vis_len = images.shape[-1]//model_div
            vis_pad_mask = torch.arange(vis_len).repeat((images.shape[0], 1))

            for b in range(images.shape[0]):
                vis_pad_mask[b] = vis_pad_mask[b] < (vis_len - vis_pad_lens[b]//model_div)
            vis_pad_mask = vis_pad_mask.unsqueeze(0)
            masked_cross_attn = vis_pad_mask.permute(1, 0, 2) * cross_attn * space_mask.T.unsqueeze(-1)
            max_attn = torch.max(masked_cross_attn, dim=-1)
            max_attn = max_attn.indices.masked_fill(max_attn.values == 0, -1)
            for batch, vis_indxs in enumerate(max_attn):
                batch_coords = []
                for vis_indx in (vis_indxs >= 0).nonzero():
                    batch_coords.append((model_div / scales[batch]) * max_attn[batch, vis_indx])
                all_coords.append(batch_coords)

        prediction_dict = dict()
        for indx in all_indices:
            if dataset.data[indx]['file_name'] not in prediction_dict:
                prediction_dict[dataset.data[indx]['file_name']] = []
            prediction_dict[dataset.data[indx]['file_name']].append(indx)

        for file_name in prediction_dict:
            pw = PageWriter()
            width = dataset.data[prediction_dict[file_name][0]]['width']
            height = dataset.data[prediction_dict[file_name][0]]['height']
            page = pw.add_page(file_name, width, height)
            tr = pw.add_text_region(page, [(0, 0), (width, 0), (width, height), (0, height)])
            for indx in prediction_dict[file_name]:
                poly_coords = dataset.data[indx]['poly']
                words, classes = utils.util.separate_string(all_predictions[indx])
                offset = 0
                custom_string = ""
                for word, word_class in zip(words.split(), classes):
                    custom_string += f"{''.join(list(word_class)[1:-1])} {{offset:{offset}; length:{len(word)}}} "
                    offset += len(word) + 1
                tl = pw.add_text_line(tr, poly_coords, custom_string.strip())
                x_ints = [x + dataset.data[indx]['crop'][0] for x in all_coords[indx]]
                y_ints = utils.util.interpolate_y(x_ints, dataset.data[indx]['baseline'])
                pw.add_base_line(tl, list(zip(x_ints, y_ints)))
                pw.add_text(tl, words)
            pw.write_xml(os.path.join('output', 'htr', file_name + '.xml'))

    def _load_checkpoint(self):
        file_name = os.path.join(self.save_dir, str(self.config.resume))
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['state_dict'])
