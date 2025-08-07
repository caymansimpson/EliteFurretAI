import os
import os.path
import re
import sys


# This file makes sure that every mon has a level 50 for VGC teams. This is because we parse
# stats from the teambuilder, and if we don't have accurate level information, we won't be able
# to parse the stats correctly
def main():
    print("Starting a script to edit all VGC teams to have level 50")
    directories = set()
    fp = sys.argv[1] if len(sys.argv) > 1 else "./data/teams"

    for format_folder in os.listdir(fp):
        folder_path = os.path.join(fp, format_folder)

        if os.path.isdir(folder_path) and "vgc" in format_folder.lower():
            print(f"Found the format: {format_folder}")
            directories.add(folder_path)

    print("Now finding all teams...")
    total = 0
    for directory in directories:
        filenames = os.listdir(directory)
        for filename in filenames:
            team_path = os.path.join(directory, filename)
            frmt = os.path.basename(directory)

            print("Found the team:", team_path)
            team_string = ""
            with open(team_path, "r") as f:
                team_string = f.read()

            team_string = re.sub(
                r"(Ability:.*\n)(Level:\s[0-9]+(\s+)?\n)?", r"\1Level: 50\n", team_string
            )

            # Copies this exact file format to your Desktop, in order to not irreversibly overwrite files
            new_path = os.path.expanduser(os.path.join("~/Desktop/", frmt, filename))
            dir_path = os.path.expanduser(os.path.join("~/Desktop/", frmt))
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            with open(new_path, "w") as f:
                f.write(team_string)

            total += 1

    print(f"Done! Cleaned up {total} teams")


if __name__ == "__main__":
    main()
