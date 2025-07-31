from PIL import Image
import os

file_name = "HoloFuloNet"

def concat_img(background_data, live_data, dead_data, save_data):
    background = Image.open(background_data)
    foreground = Image.open(live_data)
    foreground_2 = Image.open(dead_data)

    alpha = 50
    foreground = foreground.convert("RGBA")
    foreground_with_alpha = Image.new("RGBA", foreground.size)

    for x in range(foreground.width):
        for y in range(foreground.height):
            r, g, b, _ = foreground.getpixel((x, y))
            foreground_with_alpha.putpixel((x, y), (r, g, b, alpha))

    foreground_2 = foreground_2.convert("RGBA")
    foreground_2_with_alpha = Image.new("RGBA", foreground_2.size)

    for x in range(foreground_2.width):
        for y in range(foreground_2.height):
            r, g, b, _ = foreground_2.getpixel((x, y))
            foreground_2_with_alpha.putpixel((x, y), (r, g, b, alpha))

    combined = Image.alpha_composite(background.convert("RGBA"), foreground_with_alpha)
    combined = Image.alpha_composite(combined, foreground_2_with_alpha)
    combined.save(save_data)


background_folder = f"../result_mask_{file_name}/split_data/input/"
live_folder = f"../result_mask_{file_name}/watershed_live_marker_results/"
dead_folder = f"../result_mask_{file_name}/watershed_dead_marker_results"
save_folder = f"../result_mask_{file_name}/final_results/"

os.makedirs(save_folder, exist_ok=True)

for data in background_folder:
    background_data = os.path.join(background_folder, data)
    live_data = os.path.join(live_folder, data)
    dead_data = os.path.join(dead_folder, data)
    save_data = os.path.join(save_folder, data)

    concat_img(background_data, live_data, dead_data, save_data)
