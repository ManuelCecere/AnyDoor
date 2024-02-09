from collections import namedtuple
import time
import os

if __name__ == "__main__":

    Test_tuple = namedtuple(
        "Test_tuple",
        ["reference_image_path", "bg_image_path", "bg_mask_path", "save_path"],
    )

    bg_upper_dirs = [
        ["./examples/SUS/BG/Eva_0.png", "./examples/SUS/BG/Eva_mask_upper.png"],
        [
            "./examples/SUS/BG/Eva_0.png",
            "./examples/SUS/BG/Eva_mask_upper_smaller.png",
        ],
    ]
    bg_lower_dirs = [
        ["./examples/SUS/BG/Eva_0.png", "./examples/SUS/BG/Eva_mask_lower.png"],
        [
            "./examples/SUS/BG/Eva_0.png",
            "./examples/SUS/BG/Eva_mask_lower_smaller.png",
        ],
    ]
    test_paths = []
    root_directory = "./examples/SUS/FG"
    for path_tuple in bg_upper_dirs:
        for dirpath, dirnames, filenames in os.walk(
            os.path.join(root_directory, "lower")
        ):
            for filename in filenames:
                print(filename)
                print(path_tuple)
                file_path = os.path.join(dirpath, filename)  # Construct full file path
                file_id = filename.split("_")[0]
                test_tuple = Test_tuple(
                    reference_image_path=file_path,
                    bg_image_path=path_tuple[0],
                    bg_mask_path=path_tuple[1],
                    save_path=os.path.join(
                        "./examples/SUS/GEN", f"{file_id}_lower.png"
                    ),
                )
                test_paths.append(test_tuple)

    for path_tuple in bg_lower_dirs:
        for dirpath, dirnames, filenames in os.walk(
            os.path.join(root_directory, "upper")
        ):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)  # Construct full file path
                file_id = filename.split("_")[0]
                test_tuple = Test_tuple(
                    reference_image_path=file_path,
                    bg_image_path=path_tuple[0],
                    bg_mask_path=path_tuple[1],
                    save_path=os.path.join(
                        "./examples/SUS/GEN", f"{file_id}_upper.png"
                    ),
                )
                test_paths.append(test_tuple)
    print(test_paths)
