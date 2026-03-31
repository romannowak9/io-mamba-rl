# To run:
# from repo root:
# python3 -m test.afo_demo

import supervisely as sly
import json
import cv2
from utils.helpers import sort_key_from_filename


def afo_test():
    # Creating Supervisely project from local directory.
    data_dir = "data/afo"
    project = sly.Project(data_dir, sly.OpenMode.READ)

    print("Opened project: ", project.name)
    print("Number of images in project:", project.total_items)

    # Showing annotations tags and classes.
    print(project.meta)

    print("Number of classes:", len(project.meta.obj_classes))

    # Iterating over classes in project, showing their names, geometry types and colors.
    for obj_class in project.meta.obj_classes:
        print(
            f"Class '{obj_class.name}', color='{obj_class.color}', class_id={obj_class.sly_id}",
        )

    for dataset in project.datasets:
        # Iterating over images in dataset, using the paths to the images and annotations.
        for item_name, image_path, ann_path in sorted(list(dataset.items()), key=lambda x : sort_key_from_filename(x[0])):
            # print(f"Item '{item_name}': image='{image_path}', ann='{ann_path}'")
            ann_json = json.load(open(ann_path))
            ann = sly.Annotation.from_json(ann_json, project.meta)

            img = sly.image.read(image_path)  # rgb - order

            ann.draw_pretty(thickness=2)
            # lub 
            # ann.draw(img, fill_rectangles=False, thickness=2, draw_class_names=True)

            cv2.imshow("Annotated", cv2.resize(img, [img.shape[1]//5, img.shape[0]//5]))
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

            # break  # Only first sample

        break  # only train dataset
        

if __name__ == '__main__':
    afo_test()
    cv2.destroyAllWindows()