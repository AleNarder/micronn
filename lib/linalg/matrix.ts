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
        this.rows_   = rows;
        this.cols_   = cols;
        this.values_ = Array.from({ length: rows }, () => new Float64Array(cols));
    }
    // <<<<<<<<<<<<<<<<<<<<<<


    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Instance methods

    @bind
    public toArray(): number[][] {
        return this.values_.map(row => Array.from(row));
    }

    @bind
    public sum (): number {
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
    public rand(): void {
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                this.set(i, j, Math.random() * 2 - 1);
            }
        }
    }

    @bind
    public add (other: Matrix) {
        const matrix = new Matrix(this.rows_, this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j) + other.get(i, j));
            }
        }
        return matrix;
    }

    @bind
    public sub (scalar: number) {
        return this.mul(-scalar);
    }

    @bind
    public mul (scalar: number) {
        const matrix = new Matrix(this.rows_, this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j) * scalar);
            }
        }
        return matrix;
    }

    @bind
    public dot (other: Vector): Vector {
        const vector = new Vector(this.rows_);
        for (let i = 0; i < this.rows_; i++) {
            let sum = 0;
            for (let j = 0; j < this.cols_; j++) {
                sum += this.get(i, j) * other.get(j);
            }
            vector.set(i, sum);
        }
        return vector
    }

    @bind
    public inner (other: Matrix): Matrix {
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

    @bind
    public T  () {
        const matrix = new Matrix(this.cols_, this.rows_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(j, i, this.get(i, j));
            }
        }
        return matrix;
    }
}