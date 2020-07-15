from text_recognizer.datasets.iam_paragraphs_dataset import IamParagraphsDataset

if __name__ == "__main__":
    data = IamParagraphsDataset()
    data.load_or_generate_data()
    print(data)