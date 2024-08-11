import { Matrix } from './matrix';

export class Vector{

    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Static methods

    static fromArray(array: number[]): Vector {
        const vector = new Vector(array.length);
        for (let i = 0; i < array.length; i++) {
            vector.set(i, array[i]);
        }
        return vector;
    }

    static fromMatrix(matrix: Matrix): Vector {
        const vector = new Vector(matrix.cols_);
        for (let i = 0; i < matrix.cols_; i++) {
            vector.set(i, matrix.get(0, i));
        }
        return vector;
    }
    // <<<<<<<<<<<<<<<<<<<<<<
    
    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Instance properties

    readonly values_: number[];

    constructor(size: number) {
        this.values_ = new Array(size);
    }
    // <<<<<<<<<<<<<<<<<<<<<<


    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Instance methods

    get size() {
        return this.values_.length;
    }
    
    get(index) {
        return this.values_[index];
    }

    max (): number {
        let max = this.get(0);
        for (let i = 1; i < this.size; i++) {
            max = Math.max(max, this.get(i));
        }
        return max
    }
    
    set(index, value): void {
        this.values_[index] = value;
    }

    sum (): number {
        let sum = 0;
        for (let i = 0; i < this.size; i++) {
            sum += this.get(i);
        }
        return sum;
    }

    log (): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, Math.log(this.get(i)));
        }
        return vector;
    }

    exp (): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, Math.exp(this.get(i)));
        }
        return vector;
    }

    pow (exponent: number): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            this.set(i, Math.pow(this.get(i), exponent));
        }
        return vector;
    }

    T(): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, this.get(i));
        }
        return vector;
    }


    rand() {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, Math.random() * 2 - 1);
        }
        return vector;
    }

    add (other: Vector): Vector;
    add(other: number): Vector;
    add(other: number | Vector): Vector {
        if (other instanceof Vector) {
           return this._addVector(other);
        } else {
           return this._addScalar(other);
        }
    }

    private _addVector(vector: Vector): Vector {
        const result = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            result.set(i, this.get(i) + vector.get(i));
        }
        return result;
    }

    private _addScalar(scalar: number): Vector {
        const result = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            result.set(i, this.get(i) + scalar);
        }
        return result;
    }


    sub(other: Vector): Vector;
    sub(other: number): Vector;
    sub(other: number | Vector): Vector {
        if (other instanceof Vector) {
            return this._addVector(other.mul(-1));
         } else {
            return this._addScalar(other * -1);
         }
    }

    mul(other: number): Vector;
    mul(other: Vector): Vector;
    mul(other: Vector | number): Vector {
        if (other instanceof Vector) {
            return this._mulVector(other);
        } else {
            return this._mulScalar(other);
        }
    }

    private _mulVector(vector: Vector): Vector {
        const result = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            result.set(i, this.get(i) * vector.get(i));
        }
        return result;
    }

    private _mulScalar(scalar: number): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, this.get(i) * scalar);
        }
        return vector;
    }

    
    div(other: number): Vector;
    div(other: Vector): Vector;
    div (other: Vector | number) {
        if (other instanceof Vector) {
            return this._mulVector(other.pow(-1));
        } else {
            return this._mulScalar(Math.pow(other, -1));
        }
    }
    
    dot (other: Vector): number;
    dot (other: Matrix): Vector;
    dot (other: Matrix | Vector): Vector | number {
        if (other instanceof Matrix) {
            return this._dotMatrix(other);
        } else {
            return this._dotVector(other);
        }
    }

    private _dotMatrix(matrix: Matrix): Vector {
        const vector = new Vector(matrix.cols_);
        for (let i = 0; i < matrix.cols_; i++) {
            let sum = 0;
            for (let j = 0; j < this.size; j++) {
                sum += this.get(j) * matrix.get(j, i);
            }
            vector.set(i, sum);
        }
        return vector;
    }
    
    private _dotVector(vector: Vector): number {
        let sum = 0;
        for (let i = 0; i < this.size; i++) {
            sum += this.get(i) * vector.get(i);
        }
        return sum;
    }
    

    toArray(): number[] {
        return [...this.values_]
    }

    copy() {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, this.get(i));
        }
        return vector;
    }
}