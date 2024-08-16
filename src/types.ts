export interface Invertible<T> {
    forward (input: T, ...args: any): any;
    backward (outputError: T, ...args: any): T;
}