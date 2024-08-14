import bind from 'bind-decorator';
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

    static fromScalar(scalar: number): Vector {
        const v =  new Vector(1)
        v.set(0, scalar);
        return v;
    }
    // <<<<<<<<<<<<<<<<<<<<<<
    
    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Instance properties

    readonly values_: number[];

    constructor(size: number) {
        this.values_ = new Array(size).fill(0);
    }
    // <<<<<<<<<<<<<<<<<<<<<<


    // >>>>>>>>>>>>>>>>>>>>>>
    // >> Instance methods


    get size() {
        return this.values_.length;
    }
    
    @bind
    get(index: number): number {
        return this.values_[index];
    }

    @bind
    norm (): number {
        return Math.sqrt(this.dot(this));
    }

    @bind
    max (): number {
        let max = this.get(0);
        for (let i = 1; i < this.size; i++) {
            max = Math.max(max, this.get(i));
        }
        return max
    }
    
    @bind
    set(index: number, value: number): void {
        this.values_[index] = value;
    }

    @bind
    sum (): Vector {
        let sum = 0;
        for (let i = 0; i < this.size; i++) {
            sum += this.get(i);
        }
        return Vector.fromArray([sum]);
    }

    @bind
    log (): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, Math.log(this.get(i)));
        }
        return vector;
    }

    @bind
    exp (): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, Math.exp(this.get(i)));
        }
        return vector;
    }

    @bind
    pow (exponent: number): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, Math.pow(this.get(i), exponent));
        }
        return vector;
    }

    @bind
    T(): Vector {
        return this.copy();
    }

    @bind
    rand(): void {
        for (let i = 0; i < this.size; i++) {
            this.set(i, Math.random() * 2 - 1);
        }
    }

    add (other: Vector): Vector;
    add(other: number): Vector;
    @bind
    add(other: number | Vector): Vector {
        if (other instanceof Vector) {
           return this._addVector(other);
        } else {
           return this._addScalar(other);
        }
    }

    @bind
    private _addVector(vector: Vector): Vector {
        const result = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            result.set(i, this.get(i) + vector.get(i));
        }
        return result;
    }

    @bind
    private _addScalar(scalar: number): Vector {
        const result = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            result.set(i, this.get(i) + scalar);
        }
        return result;
    }


    sub(other: Vector): Vector;
    sub(other: number): Vector;
    @bind
    sub(other: number | Vector): Vector {
        
        if (other instanceof Vector) {
            return this._addVector(other.mul(-1));
         } else {
            return this._addScalar(other * -1);
         }
    }

    mul(other: number): Vector;
    mul(other: Vector): Vector;
    @bind
    mul(other: Vector | number): Vector {
        if (other instanceof Vector) {
            return this._mulVector(other);
        } else {
            return this._mulScalar(other);
        }
    }

    @bind
    private _mulVector(vector: Vector): Vector {
        const result = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            result.set(i, this.get(i) * vector.get(i));
        }
        return result;
    }

    @bind
    private _mulScalar(scalar: number): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, this.get(i) * scalar);
        }
        return vector;
    }

    
    div(other: number): Vector;
    div(other: Vector): Vector;
    @bind
    div (other: Vector | number) {
        if (other instanceof Vector) {
            return this._mulVector(other.pow(-1));
        } else {
            return this._mulScalar(Math.pow(other, -1));
        }
    }
    
    dot (other: Vector): number;
    dot (other: Matrix): Vector;
    @bind
    dot (other: Matrix | Vector): Vector | number {
        if (other instanceof Matrix) {
            return this._dotMatrix(other);
        } else {
            return this._dotVector(other);
        }
    }

    @bind
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
    
    @bind
    private _dotVector(vector: Vector): number {
        let sum = 0;
        for (let i = 0; i < this.size; i++) {
            sum += this.get(i) * vector.get(i);
        }
        return sum;
    }

    apply (fn: (x: number) => number): Vector {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, fn(this.get(i)));
        }
        return vector;
    }
    
    @bind
    toArray(): number[] {
        return [...this.values_]
    }
    
    @bind
    copy() {
        const vector = new Vector(this.size);
        for (let i = 0; i < this.size; i++) {
            vector.set(i, this.get(i));
        }
        return vector;
    }
}