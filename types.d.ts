interface Invertible<T> {
    forward (input: T, ...args: any): T;
    backward (outputError: T, ...args: any): T;
}