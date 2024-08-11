import { Vector } from "./vector";

export class Matrix {

    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Static methods

    public static fromArray(array: number[]): Matrix {
        const rows = array.length;
        const cols = 1;
        const matrix = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            matrix.set(i, 0, array[i]);
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
    public values_: number[];

    constructor(rows: number, cols: number) {
        this.rows_   = rows;
        this.cols_   = cols;
        this.values_ = new Array(rows * cols).fill(0);
    }
    // <<<<<<<<<<<<<<<<<<<<<<


    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Instance methods

    public toArray(): number[][] {
        const array: number[][] = [];
        
        for (let i = 0; i < this.rows_; i++) {
            const col = new Array(this.cols_);
            for (let j = 0; j < this.cols_; j++) {
                col.push(this.get(i, j));
            }
            array.push(col);
        }

        return array;
    }

    public get(row: number, col: number): number {
        return this.values_[row * this.cols_ + col];
    }

    public set(row: number, col: number, value: number): void {
        this.values_[row * this.cols_ + col] = value;
    }

    public randomize(): void {
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                this.set(i, j, Math.random() * 2 - 1);
            }
        }
    }

    public add (other: Matrix) {
        const matrix = new Matrix(this.rows_, this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j) + other.get(i, j));
            }
        }
        return matrix;
    }


    public sub (scalar: number) {
        return this.mul(-scalar);
    }

 
    public mul (scalar: number) {
        const matrix = new Matrix(this.rows_, this.cols_);
        for (let i = 0; i < this.rows_; i++) {
            for (let j = 0; j < this.cols_; j++) {
                matrix.set(i, j, this.get(i, j) * scalar);
            }
        }
        return matrix;
    }



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