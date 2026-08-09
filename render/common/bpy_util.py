import bpy
import math


def normalize_scale(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # set origin to the center of the bounding box
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    obj.location.x = 0
    obj.location.y = 0
    obj.location.z = 0

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    max_vert_dist = math.sqrt(max([v.co.dot(v.co) for v in obj.data.vertices]))

    for v in obj.data.vertices:
        v.co /= max_vert_dist

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
