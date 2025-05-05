echo "downloading HumanML3D_new"

git clone https://huggingface.co/datasets/NamYeongCho/HumanML3D_new
mv HumanML3D_new dataset/HumanML3D

unzip dataset/HumanML3D/new_joint_vecs.zip -d dataset/HumanML3D
unzip dataset/HumanML3D/new_joints.zip    -d dataset/HumanML3D
unzip dataset/HumanML3D/texts.zip         -d dataset/HumanML3D
