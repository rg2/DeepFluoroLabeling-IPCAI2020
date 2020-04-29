# Helper functions for overlaying annotations onto images
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from PIL import ImageFont

def get_font(font_size):
    # try and get a nice font for the overlays
    font = None
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        pass
    return font

def draw_text(draw_obj, s, pos, color_str='yellow', font_size=12):
    draw_obj.text((pos[0], pos[1]), s, fill=color_str, font=get_font(font_size))

def get_box(x, box_radius=2):
    return [(x[0] - box_radius, x[1] - box_radius),
            (x[0] + box_radius, x[1] + box_radius)]

def draw_gt_land(draw_obj, x, color_str='yellow', r=2):
    draw_obj.ellipse(get_box(x,r), fill=color_str)

def draw_circle(draw_obj, x, color_str='yellow', r=2):
    draw_obj.ellipse(get_box(x,r), fill=None, outline=color_str)

def draw_est_land(draw_obj, x, color_str='yellow', r=6):
    draw_obj.line([(x[0], x[1] + r), (x[0], x[1] - r)], fill=color_str)
    draw_obj.line([(x[0] - r, x[1]), (x[0] + r, x[1])], fill=color_str)

def draw_line(draw_obj, x, y, color_str='yellow'):
    draw_obj.line([(x[0], x[1]), (y[0], y[1])], fill=color_str)

def overlay_seg(src_img, seg, alpha, is_multi_class, num_seg_classes):
    label_colors = [ [0.0, 1.0, 0.0],  # pelvis green
                     [1.0, 0.0, 0.0],  # left femur red
                     [0.0, 0.0, 1.0],  # right femur blue
                     [1.0, 1.0, 0.0],  # yellow
                     [0.0, 1.0, 1.0],  # cyan
                     [1.0, 0.5, 0.0],  # orange
                     [0.5, 0.0, 0.5]]  # purple

    assert(num_seg_classes <= len(label_colors))

    img = src_img.clone()

    if is_multi_class:
        assert(num_seg_classes == seg.shape[0])

        # loop over RGB
        for c in range(3):
            img_c = img[c,:,:].clone()

            img_c *= 1 - alpha
            img_c += alpha * seg[0,:,:] * img[c,:,:]

            for l in range(1,num_seg_classes):
                img_c += (alpha * label_colors[l-1][c]) * seg[l,:,:]

            img[c,:,:] = img_c
    else:
        for l in range(1,num_seg_classes):
            s_idx = seg == l
            
            label_color = label_colors[l - 1]
            
            for c in range(3):
                img_c = img[c,:,:]

                img_c[s_idx] = ((1 - alpha) * img_c[s_idx]) + (alpha * label_color[c])
    
    return img
