  NAME=$1

# Load ISIC 2016 dataset
if [ "$NAME" == "ISIC2016" ]; then
  TRAIN_DATA_STRING="Training_Data"
  TRAIN_GT_STRING="Training_GroundTruth"
  TEST_DATA_STRING="Test_Data"
  TEST_GT_STRING="Test_GroundTruth"
  #
  mkdir -p "./data/$NAME"
  mkdir -p "./data/$NAME/$TRAIN_DATA_STRING"
  mkdir -p "./data/$NAME/$TRAIN_GT_STRING"
  mkdir -p "./data/$NAME/$TEST_DATA_STRING"
  mkdir -p "./data/$NAME/$TEST_GT_STRING"
  #
  ZIP_FILENAME1="ISBI2016_ISIC_Part1_$TRAIN_DATA_STRING"
  ZIP_FILENAME2="ISBI2016_ISIC_Part1_$TRAIN_GT_STRING"
  ZIP_FILENAME3="ISBI2016_ISIC_Part1_$TEST_DATA_STRING"
  ZIP_FILENAME4="ISBI2016_ISIC_Part1_$TEST_GT_STRING"
  CSV_FILENAME1="ISBI2016_ISIC_Part1_$TRAIN_GT_STRING.csv"
  CSV_FILENAME2="ISBI2016_ISIC_Part1_$TEST_GT_STRING.csv"
  #
  if [ ! -e "$ZIP_FILENAME1.zip" ]; then
    echo "Downloading $ZIP_FILENAME1.zip..."
    URL="https://isic-challenge-data.s3.amazonaws.com/2016/$ZIP_FILENAME1.zip"
    wget -N "$URL" "$ZIP_FILENAME1.zip"
  fi
  unzip "$ZIP_FILENAME1.zip" -d "./data/$NAME/"
  cp -r "./data/$NAME/$ZIP_FILENAME1/." "./data/$NAME/$TRAIN_DATA_STRING"
  rm -rf "./data/$NAME/$ZIP_FILENAME1"
  rm "$ZIP_FILENAME1.zip"
  #
  if [ ! -e "$ZIP_FILENAME2.zip" ]; then
    echo "Downloading $ZIP_FILENAME2.zip..."
    URL="https://isic-challenge-data.s3.amazonaws.com/2016/$ZIP_FILENAME2.zip"
    wget -N "$URL" "$ZIP_FILENAME2.zip"
  fi
  unzip "$ZIP_FILENAME2.zip" -d "./data/$NAME/"
  cp -r "./data/$NAME/$ZIP_FILENAME2/." "./data/$NAME/$TRAIN_GT_STRING"
  rm -rf "./data/$NAME/$ZIP_FILENAME2"
  rm "$ZIP_FILENAME2.zip"
  #
  if [ ! -e "$ZIP_FILENAME3.zip" ]; then
    echo "Downloading $ZIP_FILENAME3.zip..."
    URL="https://isic-challenge-data.s3.amazonaws.com/2016/$ZIP_FILENAME3.zip"
    wget -N "$URL" "$ZIP_FILENAME3.zip"
  fi
  unzip "$ZIP_FILENAME3.zip" -d "./data/$NAME/"
  cp -r "./data/$NAME/$ZIP_FILENAME3/." "./data/$NAME/$TEST_DATA_STRING"
  rm -rf "./data/$NAME/$ZIP_FILENAME3"
  rm "$ZIP_FILENAME3.zip"
  #
  if [ !  -e "$ZIP_FILENAME4.zip" ]; then
    echo "Downloading $ZIP_FILENAME4.zip..."
    URL="https://isic-challenge-data.s3.amazonaws.com/2016/$ZIP_FILENAME4.zip"
    wget -N "$URL" "$ZIP_FILENAME4.zip"
  fi
  unzip "$ZIP_FILENAME4.zip" -d "./data/$NAME/"
  cp -r "./data/$NAME/$ZIP_FILENAME4/." "./data/$NAME/$TEST_GT_STRING"
  rm -rf "./data/$NAME/$ZIP_FILENAME4"
  rm "$ZIP_FILENAME4.zip"
  #
  echo "Downloading $CSV_FILENAME1..."
  URL="https://raw.githubusercontent.com/MedicineToken/MedSegDiff/refs/heads/master/data/isic_csv/$CSV_FILENAME1"
  wget -N "$URL" "./data/$NAME/$CSV_FILENAME1"
  #
  echo "Downloading $CSV_FILENAME2..."
  URL="https://raw.githubusercontent.com/MedicineToken/MedSegDiff/refs/heads/master/data/isic_csv/$CSV_FILENAME2"
  wget -N "$URL" -O "./data/$NAME/$CSV_FILENAME2"

# Download REFUGE dataset
elif [ "$NAME" == "REFUGE" ]; then
  mkdir -p "./data/$NAME"
  #
  FILENAME="REFUGE-Multirater"
  #
  if [ ! -e "$FILENAME.zip" ]; then
    echo "Downloading $FILENAME.zip..."
    URL="https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/resolve/main/$FILENAME.zip"
    wget -N "$URL" -O "$FILENAME.zip"
  fi
  unzip "$FILENAME.zip" -d "./data"
  cp -r "./data/$FILENAME/." "./data/$NAME"
  rm "$FILENAME.zip"
  rm -rf "./data/$FILENAME"
  rm -rf "./data/__MACOSX"

# Download DDTI dataset
elif [ "$NAME" == "DDTI" ]; then
  FILENAME="thyroidultrasound"
  if [ ! -e "$FILENAME.zip" ]; then
    echo "Downloading $FILENAME.zip..."
    kaggle datasets download "eiraoi/$FILENAME"
  fi
  unzip "$FILENAME.zip" -d "./data/$NAME"
  rm "$FILENAME.zip"

# Download STARE dataset
elif [ "$NAME" == "STARE" ]; then
  mkdir -p "./data/$NAME/images"
  mkdir -p "./data/$NAME/labels"
  #
  IMAGE_FILENAME="stare-images"
  LABEL_FILENAME="labels-ah"
  #
  if [ ! -e "$IMAGE_FILENAME.tar" ]; then
    echo "Downloading $IMAGE_FILENAME.tar..."
    URL="https://cecas.clemson.edu/~ahoover/stare/probing/$IMAGE_FILENAME.tar"
    wget -N "$URL" "$IMAGE_FILENAME.tar"
  fi
  tar -xf "$IMAGE_FILENAME.tar" -C "./data/$NAME/images"
  for file in "./data/$NAME/images"/*; do
    if [ -f "$file" ]; then
        gzip -d "$file"
    fi
  done
  rm "$IMAGE_FILENAME.tar"
  #
  if [ ! -e "$LABEL_FILENAME.tar" ]; then
    echo "Downloading $LABEL_FILENAME.tar..."
    URL="https://cecas.clemson.edu/~ahoover/stare/probing/$LABEL_FILENAME.tar"
    wget -N "$URL" -O "$LABEL_FILENAME.tar"
  fi
  tar -xf "$LABEL_FILENAME.tar" -C "./data/$NAME/labels"
  for file in "./data/$NAME/labels"/*; do
    if [ -f "$file" ]; then
        gzip -d "$file"
    fi
  done
  rm "$LABEL_FILENAME.tar"

# Download IDRiD dataset
elif [ "$NAME" == "IDRiD" ]; then
  if [ ! -e "$NAME.zip" ]; then
    echo "$NAME.zip not found!"
  fi
  unzip "$NAME.zip" -d "./data"

# Download LiTS17 dataset
elif [ "$NAME" == "LiTS17" ]; then
  mkdir -p "./data/$NAME"
  FILENAME1="liver-tumor-segmentation"
  FILENAME2="liver-tumor-segmentation-part-2"

  if [ ! -e "$FILENAME1.zip" ]; then
    echo "Downloading $FILENAME1.zip..."
    kaggle datasets download "andrewmvd/$FILENAME1"
  fi
  if [ ! -e "$FILENAME2.zip" ]; then
    echo "Downloading $FILENAME2.zip..."
    kaggle datasets download "andrewmvd/$FILENAME2"
  fi
  unzip "$FILENAME1.zip" -d "./data/LiTS17"
  unzip "$FILENAME2.zip" -d "./data/LiTS17"
  rm "$FILENAME1.zip"
  rm "$FILENAME2.zip"
  cp -r "./data/LiTS17/segmentations" "./data/LiTS17/labels_nii"
  rm -rf "./data/LiTS17/segmentations"
  for i in {1..8}; do
    if [ "$i" -eq 7 ]; then
      continue
    fi
    cp -r "./data/LiTS17/volume_pt$i/." "./data/LiTS17/volumes_nii"
    rm -rf "./data/LiTS17/volume_pt$i"
  done

# Download FLARE dataset
elif [ "$NAME" == "FLARE22" ]; then
  mkdir -p "./data/$NAME"
  #
  FILENAME="miccai-flare22-challenge-dataset"
  #
  if [ ! -e "$FILENAME.zip" ]; then
    echo "Downloading $FILENAME.zip..."
    kaggle datasets download "prathamkumar0011/$FILENAME"
  fi
  unzip "$FILENAME.zip" -d "./data"
  cp -r "./data/${NAME}Train/${NAME}Train/images/." "./data/$NAME/images_nii"
  cp -r "./data/${NAME}Train/${NAME}Train/labels/." "./data/$NAME/labels_nii"
  rm -rf "./data/${NAME}Train"
  rm "$FILENAME.zip"
fi
