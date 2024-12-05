from PIL import Image
import glob
from PIL import ImageDraw, ImageFont

#Add text to the individual images
text_data = ['1966-1970','1971-1975','1976-1980','1981-1985','1986-1990','1991-1995','1996-2000','2001-2005','2006-2010','2011-2015','2016-2020']
text_position = (10, 10)
text_color = 'black'
font_size = 15

def stich_images_as_gif(inp_path, out_path):
    '''
    Stiches images given in the inp_path and forms a gif out of it and output gif is populated at out_path
    '''
    assert(isinstance(inp_path, str) and isinstance(out_path, str))

    paths = glob.glob(inp_path + r'\\*')
    fileNames = glob.glob(paths[0] + '\*')
    crop_box = (0, 100, 1000, 500)
    gif_images = []

    for file_val in fileNames:
        img_path_list = []
        for folders in paths:
            file_val = file_val.split('\\')[-1]
            img_path_list.append(folders + '\\' + file_val)

        imgs = []
        crop_imgs = []
        for val in img_path_list:
            img = Image.open(val)
            crop_imgs.append(img.crop(crop_box))
        
        indv_width = crop_box[2] - crop_box[0]
        indv_height = crop_box[3] - crop_box[1]

        new_width = 1000 * 3 
        new_height = 400 * 2

        matrix_img = Image.new('RGB', (new_width, new_height))

        matrix_img.paste(crop_imgs[0], (0, 0))
        matrix_img.paste(crop_imgs[1], (indv_width, 0))
        matrix_img.paste(crop_imgs[2], (indv_width*2, 0))
        matrix_img.paste(crop_imgs[3], (0, indv_height))
        matrix_img.paste(crop_imgs[4], (indv_width, indv_height))
        matrix_img.paste(crop_imgs[5], (indv_width*2, indv_height))
        gif_images.append(matrix_img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()


    def add_text(image, position, text, color, font):
        draw = ImageDraw.Draw(image)
        draw.text(position, text, fill=color, font=font)

    frames = []
    count = 0
    for image_file in gif_images:
        add_text(image_file, text_position, text[count], text_color, font)
        count += 1

    gif_images[0].save(
        out_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=2000,
        loop=0
    )