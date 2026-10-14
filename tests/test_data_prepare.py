import zipfile
import pytest
from q3_biqae.data_prepare import parse_french, split_asof

def test_french_first_table_units_missing_and_asof(tmp_path):
    text = 'Metadata\n,Industry1,Industry2\n20240101,1,2\n20240102,-99.99,2\n20240103,-2,4\n20240104,3,5\n\n,Industry1,Industry2\n20240101,99,99\n'
    path = tmp_path/'french.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('french.csv', text)
    frame = parse_french(path)
    assert len(frame) == 3
    assert frame.iloc[0, 0] == pytest.approx(.01)
    train, test = split_asof(frame, '2024-01-03', '2024-01-04')
    assert len(train) == 2 and len(test) == 1
    with pytest.raises(ValueError):
        split_asof(frame, '2024-01-04', '2024-01-04')

