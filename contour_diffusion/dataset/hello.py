'''y = ac(y).reshape(128, 128)
        y = y.cpu()
        y = y.detach().numpy()
        image = np.where(y > 0.8, 1, 0)
        binary_mask_8bit = ((image.astype(np.float32) / np.max(image)) * 255).astype(np.uint8)
        cv2.imwrite('pre_mask.png', binary_mask_8bit)

        mask = mask.reshape(128, 128)
        mask = mask.cpu()
        mask = mask.detach().numpy()
        image = np.where(mask > 0.8, 1, 0)
        binary_mask_8bit = ((image.astype(np.float32) / np.max(image)) * 255).astype(np.uint8)
        cv2.imwrite('true_mask.png', binary_mask_8bit)'''