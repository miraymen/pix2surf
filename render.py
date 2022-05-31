# run as blender -b fig.blend -P blender_multiple.py

# blender --background --python render.py -- --up_mesh arg1 --up_text arg2 --low_mesh arg3  and other arguments
import bpy, sys, os
import numpy as np
import argparse


class Renderer():
    def __init__(self):
        self.get_args()
        self.make_dirs()
        self.setup_scene()
        self.setup_lights()
        self.setup_camera()
        self.setup_objects()

    def get_args(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--up_mesh", type = str, help = "location of upper mesh")
        ap.add_argument("--up_text", type = str, help = "location of upper texture")

        ap.add_argument("--low_mesh", type = str, help = "location of lower mesh")
        ap.add_argument("--low_text", type = str, help = "location of lower text")

        ap.add_argument("--body_mesh", type = str, help = "location of body mesh")
        ap.add_argument("--body_text", type = str, help = "location of body texture")

        ap.add_argument("--scene_width", type = int, default = 175, help = "scene width")
        ap.add_argument("--scene_height", type = str, default = 350, help = "scene height")

        ap.add_argument("--total_frame", type = int, default = 90, help = "Total frames to render")
        ap.add_argument("--renderfolder", type = str, help = "Location of rendering folder")

        print(ap)
        self.args = ap.parse_args(self.extract_args())
        print(self.args.body_mesh)

    def make_dirs(self):
        if not os.path.isdir(self.args.renderfolder):
            os.makedirs(self.args.renderfolder)

    def setup_scene(self):
        self.scene = bpy.context.scene
        bpy.context.scene.render.resolution_x = self.args.scene_width
        bpy.context.scene.render.resolution_y = self.args.scene_height


        bpy.context.scene.render.image_settings.quality = 100

        self.scene.render.resolution_percentage = 100
        self.scene.render.use_border = False
        self.scene.render.alpha_mode = 'TRANSPARENT'


        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)



    def setup_lights(self):
        # Create new lamp datablock
        lamp_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')
        lamp_data.energy = 1
        # Create new object with our lamp datablock
        lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
        # Link lamp object to the scene so it'll appear in this scene
        self.scene.objects.link(lamp_object)
        # Place lamp to a specified location
        lamp_object.location = (15.0, 0.0, 15.0)
        # And finally select it make active
        lamp_object.select = True
        self.scene.objects.active = lamp_object


    def setup_camera(self):
        bpy.ops.object.camera_add()
        self.camera = bpy.data.objects['Camera']
        # Add camera to scene
        bpy.context.scene.camera = self.camera

    def insert_object(self, gar_loc, tex_loc, body_bool):
        bpy.ops.import_scene.obj(filepath=gar_loc)
        obj_object = bpy.context.selected_objects[0]  #### Imported objected gets id 0

        obj_object.scale = (4.0, 4.0, 4.0)
        obj_object.rotation_euler[1] = 0

        #Texture
        img = bpy.data.images.load(tex_loc)
        cTex = bpy.data.textures.new('Texture', type='IMAGE')
        cTex.image = img  # setup texture
        mat = bpy.data.materials.new(name='object')  # setup material

        # add texture to material
        mtex = mat.texture_slots.add()
        mtex.texture = cTex

        # Texture properties
        mat.specular_color = (1, 1, 1)
        mat.use_shadeless = True
        obj_object.data.materials.append(mat)



    def setup_objects(self):
        dirs = ['low', 'up', 'body']
        for val in dirs:
            mesh_pth = getattr(self.args, val+'_mesh')
            text_pth = getattr(self.args, val + '_text')
            body_bool = False if val == 'body' else True
            self.insert_object(mesh_pth, text_pth, body_bool)

    def extract_args(self, input_argv=None):
        """
        Pull out command-line arguments after "--". Blender ignores command-line flags
        after --, so this lets us forward command line arguments from the blender
        invocation to our own script.
        """
        if input_argv is None:
            input_argv = sys.argv
            print(input_argv)
        output_argv = []
        if '--' in input_argv:
            idx = input_argv.index('--')
            print(idx)
            output_argv = input_argv[(idx + 1):]
            print(output_argv)
        return output_argv

    def run(self):

        radius = 9.0
        for i in range(self.args.total_frame):
            # Set camera angle via parent
            import math
            self.scene.camera.location.x = radius * math.cos((i / self.args.total_frame) * (2 * np.pi))
            self.scene.camera.location.y = radius * math.sin((i / self.args.total_frame) * (2 * np.pi))
            self.scene.camera.rotation_mode = 'XYZ'
            self.scene.camera.location.z = -1.2


            self.scene.camera.rotation_euler[2] = math.atan2(self.scene.camera.location.y, self.scene.camera.location.x) + np.pi / 2
            self.scene.camera.rotation_euler[0] = np.pi / 2
            self.scene.camera.rotation_euler[1] = 0

            bpy.context.scene.render.filepath = os.path.join(self.args.renderfolder, 'im_{:02}.png'.format(i))

            bpy.ops.render.render(write_still=True)

        sys.exit()


if __name__ == "__main__":
    renderer = Renderer()
    renderer.run()

