echo "Downloading the mdm-dependencies"

git clone https://huggingface.co/datasets/NamYeongCho/mdm-dependency


mv mdm-dependency/smpl.zip body_models/smpl.zip
mv mdm-dependency/t2m.zip t2m.zip
mv mdm-dependency/glove.zip glove.zip
mv mdm-dependency/kit.zip kit.zip

rm -rf mdm-dependency
rm -rf body_models/smpl

unzip body_models/smpl.zip -d body_models
unzip t2m.zip
unzip glove.zip
unzip kit.zip

echo "Cleaning"

rm t2m.zip
rm glove.zip
rm kit.zip
rm body_models/smpl.zip