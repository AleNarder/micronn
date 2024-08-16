export function bind<T extends object, K extends keyof T>(
    instance: T,
    key: K,
    descriptor: TypedPropertyDescriptor<T[K]>
): TypedPropertyDescriptor<T[K]> | void {
    
    const originalMethod = descriptor.value;

    if (typeof originalMethod !== "function") {
        throw new Error("Can only bind methods");
    }

    return {
        ...descriptor,
        value: originalMethod.bind(instance)
    };
}
