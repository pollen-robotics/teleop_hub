import cv2  # type: ignore

# Parameters for the ChArUco board
ARUCO_DICT = cv2.aruco.DICT_4X4_1000
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_LENGTH = 20.75
MARKER_LENGTH = 15.58
LENGTH_PX = 1200  # total length of the page in pixels
MARGIN_PX = 20  # size of the margin in pixels


def generate_charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    size_ratio = SQUARES_Y / SQUARES_X
    board_img = cv2.aruco.CharucoBoard.generateImage(
        board, (LENGTH_PX, int(LENGTH_PX * size_ratio)), marginSize=MARGIN_PX
    )
    cv2.imshow("board", board_img)
    cv2.waitKey(3000)
    cv2.imwrite("charuco_board.png", board_img)


if __name__ == "__main__":
    # Generate and display the ChArUco board
    # The board will be saved as 'charuco_board.png'
    generate_charuco_board()
