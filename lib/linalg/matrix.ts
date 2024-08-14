import bind from "bind-decorator";
import { Vector } from "./vector";

export class Matrix {

    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Static methods

    public static fromArray(array: number[][]): Matrix {
        const rows = array.length;
        const cols = array[0].length;
        const matrix = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                matrix.set(i, j, array[i][j]);
            }
        }
        return matrix;
    }

    public static fromVector(vector: Vector): Matrix {
        const rows = vector.size;
        const cols = 1;
        const matrix = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            matrix.set(i, 0, vector.get(i));
        }
        return matrix;
    }
    // <<<<<<<<<<<<<<<<<<<<<<


    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Instance properties

    public readonly rows_: number;
    public readonly cols_: number;
    public values_: Float64Array[];

    constructor(rows: number, cols: number) {
        this.rows_ = rows;
        this.cols_ = cols;
        this.values_ = new Array(rows).fill(0).map(() => new Float64Array(cols).fill(0));
    }
    // <<<<<<<<<<<<<<<<<<<<<<


    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Instance methods

    @bind
    public toArray(): number[][] {
        return this.values_.map(row => Array.from(row));
    }

    @bind
    public sum(): number {
        return this.values_.reduce((acc, row) => acc + row.reduce((a, b) => a + b, 0), 0);
    }

    @bind
    public get(row: number, col: number): number {
        return this.values_[row][col];
    }

    @bind
    public set(row: number, col: number, value: number): void {
        this.values_[row][col] = value;
    }

    @bind
    public shape(): [number, number] {
        return [this.rows_, this.cols_];
    }

    @bind
    public rand(): void {
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                this.set(i, j, Math.random() - 0.5);
            }
        }
    }

    /**
     * Apply a function to each row or column
     * @param fn - the function to apply
     * @param byRow - if true, apply the function to each row, otherwise apply to each column
     * @returns - a new matrix
     */
    @bind
    public apply(fn: (value: Vector) => Vector, byRow = true): Matrix {
        const matrix = new Matrix(this.rows_, this.cols_);
        if (byRow) {
            return this._applyByRow(fn, matrix);
        } else {
            return this._applyByColumn(fn, matrix);
        }
    }

    @bind
    private _applyByRow(fn: (value: Vector) => Vector, matrix: Matrix): Matrix {
        for (let i = 0; i < this.rows_; i++) {
            const vector = new Vector(this.cols_);
            for (let j = 0; j < this.cols_; j++) {
                vector.set(j, this.get(i, j));
            }
            const result = fn(vector);
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, result.get(j));
            }
        }
        return matrix;
    }

    @bind
    private _applyByColumn(fn: (value: Vector) => Vector, matrix: Matrix): Matrix {
        for (let j = 0; j < this.cols_; j++) {
            const vector = new Vector(this.rows_);
            for (let i = 0; i < this.rows_; i++) {
                vector.set(i, this.get(i, j));
            }
            const result = fn(vector);
            for (let i = 0; i < this.rows_; i++) {
                matrix.set(i, j, result.get(i));
            }
        }
        return matrix;
    }

    /**
     * Performs element-wise addition. 
     * Performs broadcasting if needed
     * @param other - Matrix or scalar
     */
    add(other: number): Matrix;
    add(other: Matrix): Matrix;

    @bind
    public add(other: Matrix | number): Matrix {
        if (typeof other === "number") {
            return this._addScalar(other);
        } else {
            return this._addMatrix(other);
        }
    }

    @bind
    private _addMatrix(other: Matrix): Matrix {

        if (this.rows_ !== other.rows_ || this.cols_ !== other.cols_) {
            throw new Error("add failed: matrix dimensions must match");
        }

        const matrix = new Matrix(this.rows_, this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j) + other.get(i, j));
            }
        }
        return matrix;
    }

    @bind
    private _addScalar(other: number): Matrix {
        const matrix = new Matrix(this.rows_, this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j) + other);
            }
        }
        return matrix;
    }

    /**
     * Performs element-wise subtraction
     * @param other 
     */
    sub(other: number): Matrix;
    sub(other: Matrix): Matrix;
    @bind
    public sub(other: Matrix | number): Matrix {
        // Explicit type check
        if (typeof other === "number") {
            return this.add(other * -1);
        } else {
            return this.add(other.mul(-1));
        }
    }



    /**
     * Performs element-wise multiplication
     * @param other - Matrix or scalar
     */
    mul(other: Matrix): Matrix
    mul(other: number): Matrix;
    @bind
    public mul(other: Matrix | number) {
        if (typeof other === "number") {
            return this._mulScalar(other);
        } else {
            return this._mulMatrix(other);
        }
    }

    public div(other: Matrix): Matrix;
    public div(other: number): Matrix;
    @bind
    public div(other: Matrix | number): Matrix {
        if (typeof other === "number") {

            if (other === 0) {
                throw new Error("div failed: division by zero");
            }

            return this._mulScalar(1 / other);
        } else {

            if (other.values_.some(row => row.some(value => value === 0))) {
                throw new Error("div failed: division by zero");
            }

            return this._mulMatrix(other.apply(x => x.pow(-1)));
        }
    }

    @bind
    private _mulMatrix(other: Matrix): Matrix {

        if (this.rows_ !== other.rows_ || this.cols_ !== other.cols_) {
            throw new Error("mul failed: dimensions must match");
        }

        const matrix = new Matrix(this.rows_, this.cols_);

        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j) * other.get(i, j));
            }
        }
        return matrix;
    }

    @bind
    private _mulScalar(other: number): Matrix {
        const matrix = new Matrix(this.rows_, this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j) * other);
            }
        }
        return matrix;
    }


    /**
     * Performs dot product
     * @param other - Matrix or vector
     */
    dot(other: Matrix): Matrix;
    dot(other: Vector): Vector;
    @bind
    public dot(other: Matrix | Vector): Vector | Matrix {
        if (other instanceof Matrix) {
            return this._dotMatrix(other);
        } else {
            return this._dotVector(other);
        }
    }

    @bind
    private _dotVector(other: Vector): Vector {

        if (this.cols_ !== other.size) {
            throw new Error("dot failed: matrix dimensions must match");
        }

        const vector = new Vector(this.rows_);
        for (let i = 0; i < this.rows_; i++) {
            let sum = 0;
            for (let j = 0; j < this.cols_; j++) {
                sum += this.get(i, j) * other.get(j);
            }
            vector.set(i, sum);
        }
        return vector;
    }

    @bind
    private _dotMatrix(other: Matrix): Matrix {

        if (this.cols_ !== other.rows_) {
            throw new Error("dot failed: matrix dimensions must match");
        }

        const matrix = new Matrix(this.rows_, other.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < other.cols_; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols_; k++) {
                    sum += this.get(i, k) * other.get(k, j);
                }
                matrix.set(i, j, sum);
            }
        }
        return matrix;
    }

    public flatten(): Matrix {
        const matrix = new Matrix(1, this.rows_ * this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(0, i * this.cols_ + j, this.get(i, j));
            }
        }
        return matrix;
    }

    /**
     * Transpose the matrix
     * @returns - Transposed matrix
     */
    @bind
    public T() {
        const matrix = new Matrix(this.cols_, this.rows_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(j, i, this.get(i, j));
            }
        }
        return matrix;
    }

    @bind
    public broadcast(size: number, byRow = true): Matrix {
        if (byRow) {
            return this._broadcastByRow(size);
        } else {
            return this._broadcastByColumn(size);
        }
    }

    @bind
    private _broadcastByRow(rows: number): Matrix {
        const matrix = new Matrix(rows, this.cols_);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i % this.rows_, j));
            }
        }
        return matrix;
    }

    @bind
    private _broadcastByColumn(cols: number): Matrix {
        const matrix = new Matrix(this.rows_, cols);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < cols; j++) {
                matrix.set(i, j, this.get(i, j % this.cols_));
            }
        }
        return matrix;
    }

    copy(): Matrix {
        const matrix = new Matrix(this.rows_, this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j));
            }
        }
        return matrix;
    }

    print(): void {
        // Find the maximum length of any element when converted to a string
        const maxLength = Math.max(
            ...this.values_.flat().map(num => num.toString().length)
        );

        // Create a formatted string for each row
        const rows = this.values_.map(row =>
            // @ts-expect-error ts(2339)
            row.map(num => num.toString().padStart(maxLength, ' ')).join(' ')
        );

        // Join the rows with newline characters
        const result = rows.join('\n');
    }

}