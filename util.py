import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, kp_size=5, draw_border=False, colormap="gist_rainbow"):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image.data.cpu())
        image = np.transpose(image, [1, 2, 0])
        kp_array = kp_array.to(torch.float32).cpu().numpy()
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]), self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        image = torch.from_numpy(image)
        return image

    def create_image_grid(
        self,
        source,
        driving,
        generated,
        kp_source,
        kp_driving,
        transformed_frame,
        transformed_kp,
        **kwargs
    ):
        b = source.shape[0]
        
        generated = generated["prediction"].clamp(-1, 1).data.cpu()

        source = source.data.cpu()
        kp_source["fg_kp"] = kp_source["fg_kp"].data.cpu()
        src_with_kp = []
        for i in range(b):
            src_with_kp.append(
                self.draw_image_with_kp(source[i], kp_source["fg_kp"][i])
            )
        src_with_kp = torch.stack(src_with_kp).to(source.device).permute(0, 3, 1, 2)

        driving = driving.data.cpu()
        transformed_frame = transformed_frame.data.cpu()
        transformed_kp["fg_kp"] = transformed_kp["fg_kp"].data.cpu()
        driving_with_kp = []
        transformed_with_kp = []
        for i in range(b):
            driving_with_kp.append(
                self.draw_image_with_kp(driving[i], kp_driving["fg_kp"][i])
            )
            transformed_with_kp.append(
                self.draw_image_with_kp(
                    transformed_frame[i], transformed_kp["fg_kp"][i]
                )
            )

        driving_with_kp = (
            torch.stack(driving_with_kp).to(source.device).permute(0, 3, 1, 2)
        )
        transformed_with_kp = (
            torch.stack(transformed_with_kp).to(source.device).permute(0, 3, 1, 2)
        )

        img_row = [
            *map(
                lambda x: x.unsqueeze(1),
                (
                    source,
                    src_with_kp,
                    transformed_with_kp,
                    driving,
                    driving_with_kp,
                    generated,
                ),
            )
        ]
        
        if "depth" in kp_source:
            depth = kp_source["depth"].data.cpu()
            depth = F.interpolate(depth, size=source.shape[2:]).repeat(1, 3, 1, 1)
            depth = depth.unsqueeze(1)
            img_row.append(depth)

        if "occlusion_map" in kwargs:
            for i in range(len(kwargs["occlusion_map"])):
                occlusion_map = kwargs["occlusion_map"][i].to(torch.float32).data.cpu()
                occlusion_map = occlusion_map.repeat(1, 3, 1, 1)
                occlusion_map = occlusion_map
                occlusion_map = F.interpolate(occlusion_map, size=source.shape[2:])
                img_row.append(occlusion_map.unsqueeze(1))

        if "deformed_source" in kwargs:
            full_mask = []
            for i in range(kwargs['deformed_source'].shape[1]):
                image = kwargs['deformed_source'][:, i].data.cpu()
                # import ipdb;ipdb.set_trace()
                image = F.interpolate(image, size=source.shape[2:])
                mask = kwargs['contribution_maps'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[2:])

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (kwargs['deformed_source'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = torch.from_numpy(color).to(torch.float32)
                color = color.reshape((1, 3, 1, 1))
                
                img_row.append(image.unsqueeze(1))
                if i == 0:
                    img_row.append(mask.unsqueeze(1))
                else:
                    # img_row.append((torch.ones_like(mask) * color).unsqueeze(1))
                    img_row.append((mask * color).unsqueeze(1))
                    
                full_mask.append((mask * color))
            img_row.append(sum(full_mask).unsqueeze(1))

        _grid = torch.cat(img_row, dim=1)
        _grid = _grid.view(-1, *generated.shape[1:])
        grid = make_grid(
            _grid,
            nrow=len(img_row),
            normalize=True,
            value_range=(0, 1),
        )
        return grid
